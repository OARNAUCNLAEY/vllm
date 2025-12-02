# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

_CPU_ARCH_PREFER_MIXED_BATCH = (CpuArchEnum.X86, CpuArchEnum.ARM)


class CPUAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "CPU_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """CPU attention supports decoder and encoder-only attention."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_impl_cls() -> type["CPUAttentionBackendImpl"]:
        return CPUAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["CPUAttentionMetadataBuilder"]:
        return CPUAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return 2, num_blocks, num_kv_heads, block_size, head_size

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class CPUAttentionMetadata:
    isa: str
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    causal: bool = True

    # can be removed after deprecate sdpa
    use_sdpa_prefill: bool = False
    num_decode_tokens: int = 0
    sdpa_attn_masks: list[torch.Tensor | None] | None = None
    sdpa_start_loc: torch.Tensor | None = None


class CPUAttentionMetadataBuilder(AttentionMetadataBuilder[CPUAttentionMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.use_sdpa_prefill = False
        reorder_batch_threshold = None
        if current_platform.get_cpu_architecture() not in _CPU_ARCH_PREFER_MIXED_BATCH:
            # in this case, decode seqs are reordered to the front of prefill seqs
            # to split decode and prefill. Then use SDPA for prefill and
            # cpu_attention_with_kv_cache for decode
            reorder_batch_threshold = 1
            self.use_sdpa_prefill = True

        self._init_reorder_batch_threshold(reorder_batch_threshold, False)

        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config

        parallel_config = vllm_config.parallel_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(parallel_config)
        self.num_heads = vllm_config.model_config.get_num_attention_heads(
            parallel_config
        )
        self.head_dim = kv_cache_spec.head_size
        self.dtype = vllm_config.model_config.dtype
        self.window_size = getattr(kv_cache_spec, "sliding_window", -1)
        if self.window_size is None:
            self.window_size = -1
        self.block_size = vllm_config.cache_config.block_size
        self.isa = _get_attn_isa(self.dtype, self.block_size)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CPUAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        sdpa_start_loc = query_start_loc
        num_decode_tokens = 0
        if self.use_sdpa_prefill and causal:
            # Decoder, need reorder and truncate
            assert self.reorder_batch_threshold
            (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.reorder_batch_threshold,
                    require_uniform=True,
                )
            )
            num_reqs = num_decodes
            sdpa_start_loc = sdpa_start_loc[num_decodes:] - num_decode_tokens
            seq_lens = seq_lens[:num_decodes]
            query_start_loc = query_start_loc[: num_decodes + 1]
            block_table_tensor = block_table_tensor[:num_decodes]

        sheduler_metadata = None
        if causal:
            # for decode batch, use the custom kernel
            sheduler_metadata = ops.cpu_attn_get_scheduler_metadata(
                num_reqs=num_reqs,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                seq_lens=seq_lens,
                dtype=self.dtype,
                query_start_loc=query_start_loc,
                causal=causal,
                sliding_window_size=self.window_size,
                isa=self.isa,
                enable_kv_split=True,
            )

        attn_metadata = CPUAttentionMetadata(
            isa=self.isa,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            scheduler_metadata=sheduler_metadata,
            causal=causal,
            use_sdpa_prefill=self.use_sdpa_prefill,
            num_decode_tokens=num_decode_tokens,
            sdpa_start_loc=sdpa_start_loc,
        )

        return attn_metadata


class CPUAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        if logits_soft_cap is not None and attn_type in (
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        ):
            logger.warning_once(
                "CPU_ATTN does not support logits softcap for"
                " ENCODER and ENCODER_ONLY, outputs may be slightly off"
            )
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError("FP8 KV cache is unsupported in CPU_ATTN")
        self.attn_type = attn_type

        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        skip_layer: Optional[bool] = False,
    ) -> torch.Tensor:
        """Forward pass for CPU attention backend.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for CPUAttentionBackendImpl"
            )

        # For warming-up
        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            return self._run_sdpa_forward(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                self.attn_type,
            )

        # For decoder and cross-attention, use KV cache, size are
        # [num_blocks, num_kv_heads, block_size, head_size]
        key_cache, value_cache = kv_cache.unbind(0)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            ops.cpu_attn_reshape_and_cache(
                key,
                value,
            return query

        attn_type = self.attn_type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None
        if skip_layer:
            return None
        if (attn_type != AttentionType.ENCODER and kv_cache.numel() > 0):
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache, value_cache = self.paged_attn_impl.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            if (key is not None) and (value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    # During cross-attention decode, key & value will be None,
                    # preventing this IF-statement branch from running
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                self.paged_attn_impl.write_to_paged_cache(
                    key, value, key_cache, value_cache, updated_slot_mapping,
                    self.kv_cache_dtype, layer._k_scale, layer._v_scale)

        if attn_type != AttentionType.ENCODER:
            # Decoder self-attention supports chunked prefill.
            # Encoder/decoder cross-attention requires no chunked
            # prefill (100% prefill or 100% decode tokens, no mix)
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
        else:
            # Encoder attention - chunked prefill is not applicable;
            # derive token-count from query shape & and treat them
            # as 100% prefill tokens
            assert attn_metadata.num_encoder_tokens is not None
            num_prefill_tokens = attn_metadata.num_encoder_tokens
            num_decode_tokens = 0

        if attn_type == AttentionType.DECODER:
            # Only enforce this shape-constraint for decoder
            # self-attention
            assert key.shape[0] == num_prefill_tokens + num_decode_tokens
            assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        if prefill_meta := attn_metadata.prefill_metadata:
            if not prefill_meta.prefill_metadata.chunked_prefill:  # type: ignore
                assert attn_metadata.seq_lens is not None
                self._run_sdpa_forward(output,
                                       query,
                                       key,
                                       value,
                                       prefill_meta,
                                       attn_type=attn_type)
            else:
                # prefix-enabled attention
                assert not self.need_mask
                import intel_extension_for_pytorch.llm.modules as ipex_modules
                output = torch.empty_like(query)
                ipex_modules.PagedAttention.flash_attn_varlen_func(
                    output[:prefill_meta.num_prefill_tokens, :, :],
                    query[:prefill_meta.num_prefill_tokens, :, :],
                    key_cache,
                    value_cache,
                    prefill_meta.prefill_query_start_loc,
                    prefill_meta.kv_start_loc,
                    prefill_meta.max_query_len,
                    prefill_meta.max_kv_len,
                    self.scale,
                    True,
                    prefill_meta.prefill_block_tables,
                    self.alibi_slopes,
                )

        if decode_meta := attn_metadata.decode_metadata:
            assert attn_type != AttentionType.ENCODER_ONLY, (
                "Encoder-only models should not have decode metadata.")
            # Decoding run.
            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = decode_meta.get_seq_len_block_table_args(attn_type)

            self.paged_attn_impl.forward_decode(
                output[attn_metadata.num_prefill_tokens:, :, :],
                query[attn_metadata.num_prefill_tokens:, :, :],
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                attn_metadata.isa,
            )

        if attn_metadata.use_sdpa_prefill:
            assert self.sinks is None, "Attention sink is unsupported in SDPA prefill"
            num_decode_tokens = attn_metadata.num_decode_tokens
            self._run_sdpa_forward(
                query[num_decode_tokens:num_actual_tokens],
                key[num_decode_tokens:num_actual_tokens],
                value[num_decode_tokens:num_actual_tokens],
                output[num_decode_tokens:num_actual_tokens],
                attn_metadata,
                self.attn_type,
            )
            num_actual_tokens = num_decode_tokens

        if num_actual_tokens > 0:
            ops.cpu_attention_with_kv_cache(
                query=query[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                output=output[:num_actual_tokens],  # type: ignore
                query_start_loc=attn_metadata.query_start_loc,
                seq_lens=attn_metadata.seq_lens,
                scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,  # type: ignore
                sliding_window=self.sliding_window,
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                s_aux=self.sinks,
            )

        return output

    def _run_sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: CPUAttentionMetadata,
        attn_type: str,
    ) -> torch.Tensor:
        attn_masks = attn_metadata.sdpa_attn_masks
        if attn_masks is None:
            if self.alibi_slopes is not None:
                attn_masks = _make_alibi_bias(
                    self.alibi_slopes,
                    query.dtype,
                    attn_metadata.sdpa_start_loc,
                )
            elif self.sliding_window[0] != -1 or self.sliding_window[1] != -1:
                assert attn_metadata.seq_lens is not None
                attn_masks = _make_sliding_window_bias(
                    attn_metadata.sdpa_start_loc,
                    self.sliding_window[0],
                    self.sliding_window[1],
                    query.dtype,
                )
            else:
                attn_masks = [None] * (attn_metadata.sdpa_start_loc.size(0) - 1)  # type: ignore
            attn_metadata.sdpa_attn_masks = attn_masks

        query = query.movedim(0, query.dim() - 2)
        key = key.movedim(0, key.dim() - 2)
        value = value.movedim(0, value.dim() - 2)

        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=-3)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=-3)

        causal_attn = attn_type == AttentionType.DECODER

        sdpa_start_loc = attn_metadata.sdpa_start_loc.numpy()  # type: ignore
        for i in range(len(attn_masks)):
            mask = attn_masks[i]
            start_q = sdpa_start_loc[i]
            end_q = sdpa_start_loc[i + 1]
            sub_out = (
                torch.nn.functional.scaled_dot_product_attention(
                    query[None, :, start_q:end_q, :],
                    key[None, :, start_q:end_q, :],
                    value[None, :, start_q:end_q, :],
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=causal_attn and mask is None,
                    scale=self.scale,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = sub_out
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
    sdpa_start_loc: torch.Tensor,
) -> list[torch.Tensor]:
    attn_biases: list[torch.Tensor] = []
    seq_num = sdpa_start_loc.size(0) - 1
    sdpa_start_loc = sdpa_start_loc.numpy()  # type: ignore
    for i in range(seq_num):
        seq_len = sdpa_start_loc[i + 1] - sdpa_start_loc[i]
        bias = torch.arange(seq_len, dtype=dtype)  # type: ignore
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        bias = bias[None, :] - bias[:, None]

        num_heads = alibi_slopes.shape[0]
        bias = bias[None, :].repeat((num_heads, 1, 1))
        bias.mul_(alibi_slopes[:, None, None]).unsqueeze_(0)
        inf_mask = (
            torch.empty((1, seq_len, seq_len), dtype=bias.dtype)  # type: ignore
            .fill_(-torch.inf)
            .triu_(diagonal=1)
        )
        attn_biases.append((bias + inf_mask).to(dtype))

    return attn_biases


def _make_sliding_window_bias(
    sdpa_start_loc: torch.Tensor,
    left_window_size: int,
    right_window_size: int,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    attn_biases: list[torch.Tensor] = []
    seq_num = sdpa_start_loc.size(0) - 1
    sdpa_start_loc = sdpa_start_loc.numpy()  # type: ignore
    for i in range(seq_num):
        seq_len = sdpa_start_loc[i + 1] - sdpa_start_loc[i]
        mask = torch.full(  # type: ignore
            (1, seq_len, seq_len),  # type: ignore
            fill_value=1,
            dtype=dtype,
        )

        if right_window_size != -1:
            mask = torch.tril(mask, diagonal=right_window_size)
        if left_window_size != -1:
            mask = torch.triu(mask, diagonal=-left_window_size)
        mask = torch.log(mask)
        attn_biases.append(mask)

    return attn_biases


def _get_attn_isa(dtype: torch.dtype, block_size: int) -> str:
    supports_amx = torch._C._cpu._is_amx_tile_supported()
    if supports_amx and dtype in (torch.bfloat16,) and block_size % 32 == 0:
        return "amx"
    elif block_size % 32 == 0:
        if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
            return "neon"
        else:
            return "vec"
    else:
        return "vec16"
