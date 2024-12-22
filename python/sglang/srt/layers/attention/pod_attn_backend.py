from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl
from torch.nn.functional import scaled_dot_product_attention

from pod_attn import true_fused_attn_with_kvcache
from typing import TYPE_CHECKING, List

from sglang.srt.layers.attention import AttentionBackend
from sglang.global_config import global_config
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import (
    get_bool_env_var,
    is_flashinfer_available,
    should_use_tensor_core,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


class PODAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // model_runner.tp_size,
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                model_runner.tp_size
            ),
        )

        self.max_context_len = model_runner.model_config.context_len

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        # Allocate buffers
        self.workspace_buffer = torch.empty(
            global_config.flashinfer_workspace_size,
            dtype=torch.uint8,
            device=model_runner.device,
        )
        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = [
            torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device)
            for _ in range(self.num_wrappers)
        ]
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.qo_indptr = [
            torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device)
            for _ in range(self.num_wrappers)
        ]

        # Create wrappers
        # NOTE: we do not use ragged attention when there are multiple wrappers
        self.prefill_wrapper_ragged = (
            BatchPrefillWithRaggedKVCacheWrapper(self.workspace_buffer, "NHD")
            if self.num_wrappers == 1
            else None
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.decode_wrappers = []
        for _ in range(self.num_wrappers):
            self.prefill_wrappers_paged.append(
                BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
            )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        # Create indices updater
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)
        self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
            model_runner, self
        )

        # Other metadata
        self.forward_metadata = None
        self.cuda_graph_metadata = {}

        self.kv_indices = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrappers=None,
                encoder_lens=forward_batch.encoder_lens,
            )
            self.forward_metadata = (self.decode_wrappers,)
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            # Some heuristics to check whether to use ragged forward
            if forward_batch.extend_num_tokens >= 4096 and self.num_wrappers == 1:
                use_ragged = True
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            else:
                use_ragged = False
                extend_no_prefix = False

            self.kv_indices = self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
            )

            self.forward_metadata = (use_ragged, extend_no_prefix)

    def init_cuda_graph_state(self, max_bs: int):
        cuda_graph_kv_indices = torch.zeros(
            (max_bs * self.max_context_len,),
            dtype=torch.int32,
            device="cuda",
        )
        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: torch.Tensor = None,
    ):
        decode_wrappers = []
        for i in range(self.num_wrappers):
            decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=self.decode_use_tensor_cores,
                    paged_kv_indptr_buffer=self.kv_indptr[i][: bs + 1],
                    paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buffer=self.kv_last_page_len[:bs],
                )
            )

        seq_lens_sum = seq_lens.sum().item()
        self.indices_updater_decode.update(
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            decode_wrappers=decode_wrappers,
            encoder_lens=encoder_lens,
        )
        self.cuda_graph_metadata[bs] = decode_wrappers
        self.forward_metadata = (decode_wrappers,)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: torch.Tensor = None,
    ):
        self.indices_updater_decode.update(
            req_pool_indices[:bs],
            seq_lens[:bs],
            seq_lens_sum,
            decode_wrappers=self.cuda_graph_metadata[bs],
            encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0
    
    def _forward_mix(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        prev_dtype = q.dtype
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
        
        # split the q, k, v tensors into two parts: prefill and decode
        # according to the forward_batch
        # the structure of the batch is [p1, p2, ..., pN, d1, d2, ..., dN]
        # we just calculate the extend_seq_lens=1 for the decode part

        # calculate how many consistent 1 from the end of the extend_seq_lens?
        bs_d = 0
        for i in range(len(forward_batch.extend_seq_lens)):
            if forward_batch.extend_seq_lens[-1 - i] == 1:
                bs_d += 1
            else:
                break
        bs_p = len(forward_batch.extend_seq_lens) - bs_d

        cu_seq_len_q = torch.cumsum(forward_batch.extend_seq_lens, dim=0).to(q.device)
        cu_seq_len_q = torch.cat([torch.tensor([0], device=cu_seq_len_q.device), cu_seq_len_q])
        cu_seq_len_cache = torch.cumsum(forward_batch.extend_prefix_lens + forward_batch.extend_seq_lens, dim=0).to(k.device)
        cu_seq_len_cache = torch.cat([torch.tensor([0], device=cu_seq_len_cache.device), cu_seq_len_cache])

        extend_p = torch.sum(forward_batch.extend_seq_lens[:bs_p]).item()
        max_extend_p = torch.max(forward_batch.extend_seq_lens[:bs_p]).item()
        # extend_d = torch.sum(forward_batch.extend_seq_lens[bs_p:]).item()
        q_p = torch.empty((bs_p, max_extend_p, layer.tp_q_head_num, layer.head_dim), device=q.device, dtype=q.dtype)
        select_q = torch.zeros((bs_p, max_extend_p), device=q.device, dtype=torch.bool)
        
        # Keep the PyTorch Version
        # for i in range(bs_p):
        #     bs_len = forward_batch.extend_seq_lens[i]
        #     q_p[i, :bs_len] = q[cu_seq_len_q[i]:cu_seq_len_q[i+1]].reshape(bs_len, layer.tp_q_head_num, layer.head_dim)
        #     select_q[i, :bs_len] = torch.ones(bs_len, device=select_q.device)
        #     #k_p[i, :bs_len] = k[cu_seq_len_q[i]:cu_seq_len_q[i+1]].reshape(bs_len, layer.tp_k_head_num, layer.head_dim)
        #     #v_p[i, :bs_len] = v[cu_seq_len_q[i]:cu_seq_len_q[i+1]].reshape(bs_len, layer.tp_v_head_num, layer.head_dim)
        q = q.view(-1, layer.tp_q_head_num * layer.head_dim)
        q_p = q_p.view(-1, max_extend_p, layer.tp_q_head_num * layer.head_dim)
        reorder_q_kernel[(bs_p, (max_extend_p + 511) // 512)](
            q, q_p, cu_seq_len_q, select_q, 
            q.stride(0), q_p.stride(0), q_p.stride(1), select_q.stride(0)
        )
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        q_p = q_p.view(-1, max_extend_p, layer.tp_q_head_num, layer.head_dim).to(torch.float16)
        q_d = q[extend_p:].reshape(bs_d, 1, layer.tp_q_head_num, layer.head_dim).to(torch.float16)

        # then calculate the k, v tensors
        cache_seqlens = forward_batch.extend_prefix_lens + forward_batch.extend_seq_lens
        cache_seqlens_p = forward_batch.extend_prefix_lens[:bs_p] + forward_batch.extend_seq_lens[:bs_p]
        cache_seqlens_d = forward_batch.extend_prefix_lens[bs_p:] + forward_batch.extend_seq_lens[bs_p:]
        # cu_cache_seq_len = torch.cumsum(cache_seqlens, dim=0).to(k.device)
        # cu_cache_seq_len = torch.cat([torch.tensor([0], device=cu_cache_seq_len.device), cu_cache_seq_len])
        max_cache_p = torch.max(cache_seqlens_p).item()
        max_cache_d = torch.max(cache_seqlens_d).item()
        k_cache_d = torch.empty((bs_d, max_cache_d, layer.tp_k_head_num, layer.head_dim), device=k.device, dtype=torch.float16)
        v_cache_d = torch.empty((bs_d, max_cache_d, layer.tp_v_head_num, layer.head_dim), device=v.device, dtype=torch.float16)
        k_cache_p = torch.empty((bs_p, max_cache_p, layer.tp_k_head_num, layer.head_dim), device=k.device, dtype=torch.float16)
        v_cache_p = torch.empty((bs_p, max_cache_p, layer.tp_v_head_num, layer.head_dim), device=v.device, dtype=torch.float16)
        
        # Keep the PyTorch Version
        # for i in range(bs_p):
        #     bs_len = forward_batch.extend_prefix_lens[i] + forward_batch.extend_seq_lens[i]
        #     cache_seqlens_p[i] = bs_len
        #     k_cache_p[i, :bs_len] = \
        #         forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id) \
        #             [self.kv_indices[cu_cache_seq_len[i]:cu_cache_seq_len[i+1]]] \
        #                 .reshape(bs_len, layer.tp_k_head_num, layer.head_dim)
        #     v_cache_p[i, :bs_len] = \
        #         forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id) \
        #             [self.kv_indices[cu_cache_seq_len[i]:cu_cache_seq_len[i+1]]] \
        #                 .reshape(bs_len, layer.tp_k_head_num, layer.head_dim)
        
        reorder_kv_kernel[(bs_p, (max_cache_p + 511) // 512, layer.tp_k_head_num)](
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.kv_indices,
            cu_seq_len_cache[:bs_p+1],
            k_cache_p, v_cache_p,
            layer.head_dim,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).stride(0),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).stride(1),
            k_cache_p.stride(0), k_cache_p.stride(1), k_cache_p.stride(2),
        )

        # Keep the PyTorch Version
        # for i in range(bs_d):
        #     bs_len = forward_batch.extend_prefix_lens[bs_p + i] + forward_batch.extend_seq_lens[bs_p + i]
        #     # print(i, bs_len, forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id) \
        #     #         [self.kv_indices[cu_cache_seq_len[bs_p + i]:cu_cache_seq_len[bs_p + i + 1]]] \
        #     #             .reshape(bs_len, layer.tp_k_head_num, layer.head_dim).shape)
        #     cache_seqlens_d[i] = bs_len
        #     k_cache_d[i, :bs_len] = \
        #         forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id) \
        #             [self.kv_indices[cu_cache_seq_len[bs_p + i]:cu_cache_seq_len[bs_p + i + 1]]] \
        #                 .reshape(bs_len, layer.tp_k_head_num, layer.head_dim)
        #     v_cache_d[i, :bs_len] = \
        #         forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id) \
        #             [self.kv_indices[cu_cache_seq_len[bs_p + i]:cu_cache_seq_len[bs_p + i + 1]]] \
        #                 .reshape(bs_len, layer.tp_k_head_num, layer.head_dim)
        
        reorder_kv_kernel[(bs_p, (max_cache_d + 511) // 512, layer.tp_k_head_num)](
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.kv_indices,
            cu_seq_len_cache[bs_p:],
            k_cache_d, v_cache_d,
            layer.head_dim,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).stride(0),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).stride(1),
            k_cache_d.stride(0), k_cache_d.stride(1), k_cache_d.stride(2),
        )

        o_prefill, o_decode = true_fused_attn_with_kvcache(
            q_p, k_cache_p, v_cache_p, q_d, k_cache_d, v_cache_d, 
            cache_seqlens_p=cache_seqlens_p, cache_seqlens_d=cache_seqlens_d,
            softmax_scale=layer.scaling,
            causal=not layer.is_cross_attention,
            softcap=layer.logit_cap,
        )
        o_prefill = o_prefill.view(-1, layer.tp_q_head_num * layer.head_dim)
        o_decode = o_decode.view(-1, layer.tp_q_head_num * layer.head_dim)
        select_q = select_q.view(-1)
        return torch.cat([o_prefill[select_q], o_decode], dim=0).to(prev_dtype)

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if forward_batch.forward_mode.is_mixed():
            return self._forward_mix(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
        prefill_wrapper_paged = self.prefill_wrappers_paged[
            self._get_wrapper_idx(layer)
        ]

        use_ragged, extend_no_prefix = self.forward_metadata
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if not use_ragged:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            o = prefill_wrapper_paged.forward(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=not layer.is_cross_attention,
                sm_scale=layer.scaling,
                window_left=layer.sliding_window_size,
                logits_soft_cap=layer.logit_cap,
            )
        else:
            o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim),
                v.contiguous().view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
            )

            if extend_no_prefix:
                o = o1
            else:
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                )

                o, _ = merge_state(o1, s1, o2, s2)

            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata[0][self._get_wrapper_idx(layer)]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size

        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.decode_wrappers = attn_backend.decode_wrappers

        # Dispatch
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers

        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.minimum(  # TODO: replace this with clamp
                    seq_lens,
                    torch.tensor(self.sliding_window_size + 1),
                )
                paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                kv_start_idx_tmp = None

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers

        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
            else:
                # Cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
            )

    def call_begin_forward(
        self,
        wrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
    ):
        bs = len(req_pool_indices)
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            self.req_to_token.shape[1],
        )

        wrapper.end_forward()
        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            data_type=self.data_type,
            q_data_type=self.q_data_type,
        )


class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size

        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.wrapper_ragged = attn_backend.prefill_wrapper_ragged
        self.wrappers_paged = attn_backend.prefill_wrappers_paged

        # Dispatch
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        return self.call_begin_forward(
            self.wrapper_ragged,
            self.wrappers_paged[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # window attention use paged only
                paged_kernel_lens = torch.minimum(
                    seq_lens,
                    torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,
                )
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum

            kv_start_idx = seq_lens - paged_kernel_lens

            res = self.call_begin_forward(
                self.wrapper_ragged,
                self.wrappers_paged[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
            )

        return res

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            res = self.call_begin_forward(
                self.wrapper_ragged,
                self.wrappers_paged[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
            )
        return res

    def call_begin_forward(
        self,
        wrapper_ragged,
        wrapper_paged,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
    ):
        bs = len(req_pool_indices)
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            self.req_to_token.shape[1],
        )

        qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        # extend part
        if use_ragged:
            wrapper_ragged.end_forward()
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        # cached part
        wrapper_paged.end_forward()
        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
        )

        return kv_indices
        

@triton.jit
def reorder_q_kernel(
    q_ptr,              # Pointer to the input tensor q [seq_len, hidden_size]
    q_new_ptr,          # Pointer to the output tensor q_new [batch_size, max_seq_len, hidden_size]
    cu_seq_len_q_ptr,   # Pointer to cumulative sequence lengths [batch_size + 1]
    select_qp_ptr,      # Pointer to the mask tensor select_qp [batch_size, max_seq_len]
    q_stride_0,         # Stride for the first dimension of q (usually hidden_size)
    q_new_stride_0,     # Stride for the first dimension of q_new (usually max_seq_len * hidden_size)
    q_new_stride_1,     # Stride for the second dimension of q_new (usually hidden_size)
    select_qp_stride_0,  # Stride for the first dimension of select_qp (usually max_seq_len)
    BLOCK_SIZE_SEQ: tl.constexpr = 512,  # Block size for sequence length
    BLOCK_SIZE_HIDDEN: tl.constexpr = 1024,  # Block size for hidden dimension
):
    #BLOCK_SIZE_SEQ: tl.constexpr = 512
    #BLOCK_SIZE_HIDDEN: tl.constexpr = 1024

    # Batch index (each block processes one batch)
    batch_id = tl.program_id(0)
    seq_block_id = tl.program_id(1)
    hidden_size = q_stride_0

    # Load the start and end indices for the current batch
    seq_start = tl.load(cu_seq_len_q_ptr + batch_id)
    seq_end = tl.load(cu_seq_len_q_ptr + batch_id + 1)
    batch_seq_len = seq_end - seq_start  # Length of the current batch

    # Sequence indices for the current block
    seq_offset = tl.arange(0, BLOCK_SIZE_SEQ) + seq_block_id * BLOCK_SIZE_SEQ
    seq_indices = seq_start + seq_offset  # Absolute indices in q
    seq_mask = seq_offset < batch_seq_len  # Mask for valid sequences

    tl.store(
        select_qp_ptr + batch_id * select_qp_stride_0 + seq_offset,
        tl.full(seq_mask.shape, 1, dtype=tl.int8),
        mask=seq_mask,
    )
    # print(batch_id, seq_start, seq_end)
    # print(seq_mask)

    # Iterate over hidden dimension blocks
    for hidden_block_start in range(0, hidden_size, BLOCK_SIZE_HIDDEN):
        # Hidden dimension indices for the current block
        hidden_offset = tl.arange(0, BLOCK_SIZE_HIDDEN) + hidden_block_start
        hidden_mask = hidden_offset < hidden_size  # Mask for valid hidden positions

        # Check if the current block has valid indices in either dimension
        if tl.sum(seq_mask) > 0 and tl.sum(hidden_mask) > 0:  
            data = tl.load(
                q_ptr + seq_indices[:, None] * q_stride_0 + hidden_offset[None, :],
                mask=seq_mask[:, None] & hidden_mask[None, :],
                other=0.0  # Fill invalid positions with 0
            )
            tl.store(
                q_new_ptr + batch_id * q_new_stride_0 + seq_offset[:, None] * q_new_stride_1 + hidden_offset[None, :],
                data,
                mask=seq_mask[:, None] & hidden_mask[None, :]
            )
    # print(f'finish {batch_id}, {seq_block_id}')


@triton.jit
def reorder_kv_kernel(
    key_buffer_ptr,      # Pointer to key buffer [buffer_size, head_num, head_dim]
    value_buffer_ptr,    # Pointer to value buffer [buffer_size, head_num, head_dim]
    kv_indices_ptr,      # Pointer to kv_indices [sum_seq]
    cu_seq_len_cache_ptr,  # Pointer to cumulative sequence lengths [batch_size + 1]
    key_new_ptr,         # Pointer to the new key tensor [batch_size, max_seq_len, head_num, head_dim]
    value_new_ptr,       # Pointer to the new value tensor [batch_size, max_seq_len, head_num, head_dim]
    head_dim,            # Dimension of each head
    kv_buffer_stride_0,     # Stride for the first dimension of key_buffer (head_num * head_dim)
    kv_buffer_stride_1,     # Stride for the second dimension of key_buffer (head_dim)
    kv_new_stride_0,    # Stride for the first dimension of key_new (max_seq_len * head_num * head_dim)
    kv_new_stride_1,    # Stride for the second dimension of key_new (head_num * head_dim)
    kv_new_stride_2,    # Stride for the third dimension of key_new (head_dim)
    BLOCK_SIZE_SEQ: tl.constexpr = 512,  # Block size for sequence length
    BLOCK_SIZE_HEAD_DIM: tl.constexpr = 128,  # Block size for head_dim
):
    # Batch ID (each block processes one batch)
    batch_id = tl.program_id(0)
    seq_block_id = tl.program_id(1)
    head_id = tl.program_id(2)

    # Load the start and end indices for the current batch
    seq_start = tl.load(cu_seq_len_cache_ptr + batch_id)  # Start index in kv_indices
    seq_end = tl.load(cu_seq_len_cache_ptr + batch_id + 1)  # End index in kv_indices
    batch_seq_len = seq_end - seq_start  # Length of the current batch

    # Sequence indices for the current block
    seq_offset = tl.arange(0, BLOCK_SIZE_SEQ) + seq_block_id * BLOCK_SIZE_SEQ  # Sequence block offsets
    kv_indices_offset = seq_start + seq_offset  # Offset into kv_indices
    seq_mask = seq_offset < batch_seq_len  # Mask to ensure valid sequence indices

    # Process head_dim in blocks
    for head_block_start in range(0, head_dim, BLOCK_SIZE_HEAD_DIM):
        # Head dimension indices for the current block
        head_dim_offset = tl.arange(0, BLOCK_SIZE_HEAD_DIM) + head_block_start
        head_dim_mask = head_dim_offset < head_dim  # Mask for valid head_dim range

        # If valid indices exist
        if tl.sum(seq_mask, axis=0) > 0 and tl.sum(head_dim_mask, axis=0) > 0:
            # Gather indices for the current batch
            kv_indices = tl.load(
                kv_indices_ptr + kv_indices_offset,
                mask=seq_mask,
                other=0  # Fallback for invalid indices
            )

            # Load Key from buffer
            key_data = tl.load(
                key_buffer_ptr + kv_indices[:, None] * kv_buffer_stride_0
                               + head_id * kv_buffer_stride_1
                               + head_dim_offset[None, :],
                mask=seq_mask[:, None] & head_dim_mask[None, :],
                other=0.0
            )

            # Load Value from buffer
            value_data = tl.load(
                value_buffer_ptr + kv_indices[:, None] * kv_buffer_stride_0
                                 + head_id * kv_buffer_stride_1
                                 + head_dim_offset[None, :],
                mask=seq_mask[:, None] & head_dim_mask[None, :],
                other=0.0
            )

            # Store Key in new tensor
            tl.store(
                key_new_ptr + batch_id * kv_new_stride_0
                             + seq_offset[:, None] * kv_new_stride_1
                             + head_id * kv_new_stride_2
                             + head_dim_offset[None, :],
                key_data,
                mask=seq_mask[:, None] & head_dim_mask[None, :]
            )

            # Store Value in new tensor
            tl.store(
                value_new_ptr + batch_id * kv_new_stride_0
                               + seq_offset[:, None] * kv_new_stride_1
                               + head_id * kv_new_stride_2
                               + head_dim_offset[None, :],
                value_data,
                mask=seq_mask[:, None] & head_dim_mask[None, :]
            )


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)
