import time
import torch
from torch.nn.functional import cosine_similarity
import logging
import threading
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Optional, Callable
from queue import Empty, Full, Queue

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.selective_loading.query_collector import QueryCollector, SAMPLED_LAYERS

logger = logging.getLogger(__name__)

NUM_TORCH_SUBPROCESSES = 32
STATIC_KV_LEN = 128


@dataclass
class KVSelectOperation:
    rid: str
    query: torch.Tensor
    key_indices: torch.Tensor


class KVSelectEntry:
    def __init__(self, rid: str, input_ids: Tuple[int], output_ids: List[int]):
        self.rid = rid
        self.fill_ids: List[int] = list(input_ids) + output_ids
        self.originial_input_len: int = len(input_ids)
        self.accumu_length: int = 0
        self.last_compute_time: float = 0.0
        self.total_compute_time: float = 0.0
        self.accumu_attn_scores: Optional[torch.Tensor] = None
        self.compressed_indices: Optional[torch.Tensor] = None
        self.last_length: int = len(self.fill_ids)
        self.cached_queries: List[torch.Tensor] = []
        self.compress_ratio: float = 0.75
    
    def update_output_ids(self, req: Req):
        assert req.rid == self.rid, f"Request ID mismatch: {req.rid} != {self.rid}"
        num_token_new = len(req.origin_input_ids) + len(req.output_ids) - self.last_length
        if num_token_new == 0:
            return # no new tokens
        self.fill_ids.extend(req.output_ids[-num_token_new:])
        assert len(req.output_ids) >= num_token_new >= 0, \
            f"Invalid token update: {num_token_new} new tokens, " \
            f"output_ids length {len(req.output_ids)}, last_length {self.last_length}"
        self.last_length += num_token_new
    
    def restore_req(self, req: Req):
        assert req.rid == self.rid, f"Request ID mismatch: {req.rid} != {self.rid}"
        req.origin_input_ids = self.fill_ids[:self.originial_input_len]
        req.output_ids = self.fill_ids[self.originial_input_len:]
        self.last_length = len(self.fill_ids)


class KVSelector:
    def __init__(
            self, 
            head_num: int, head_dim: int, 
            kv_buffer: torch.Tensor,
            host_alloc: Callable[[int], torch.Tensor] = None,
            host_free: Callable[[torch.Tensor], int] = None,
        ):
        self.entries: Dict[str, KVSelectEntry] = {}
        self.compute_op_count: Dict[str, int] = {}
        self.finished_reqs: Set[str] = set()
        self.query_collector = QueryCollector()
        self.op_queue = Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = len(SAMPLED_LAYERS)
        # NOTE: this `detach` is important, otherwise pytorch would block other threads
        #       from accessing the key buffer
        self.kv_buffer = kv_buffer.detach()
        self.host_alloc = host_alloc
        self.host_free = host_free
    
    def reset(self):
        self.stop_event.set()
        self.worker.join()
        self.entries.clear()
        self.compute_op_count.clear()
        self.finished_reqs.clear()
        self.op_queue = Queue()
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
    
    def set_compress_ratio(self, rid: str, ratio: float) -> None:
        if entry := self.entries.get(rid):
            entry.compress_ratio = ratio
            if entry.compressed_indices is not None:
                # recompute the compressed indices
                self.host_free(entry.compressed_indices)
                entry.compressed_indices = None
        else:
            logger.warning(f"Request {rid} not found in KVSelectEntry, ignoring compress ratio update")
    
    def get_compressed_indices(self, req: Req, recent_len: int) -> Optional[torch.Tensor]:
        if entry := self.entries.get(req.rid):
            entry.update_output_ids(req)
            if entry.compressed_indices is not None:
                indices = entry.compressed_indices
                fill_ids = entry.fill_ids[:len(indices) + recent_len]
                req.origin_input_ids = fill_ids[:len(indices)]
                req.output_ids = fill_ids[-recent_len:]
                req.output_ids.append(entry.fill_ids[-1])
                entry.last_length = len(fill_ids) + 1
                return indices
        return None

    def update_with_batch(self, batch: ScheduleBatch, req_to_remove: Set[str]) -> None:
        """NOTE: The query tensors captured by QueryCollector are not associated
        with any request ID. This function matches the query tensors with the
        last batch of requests in order."""
        queries = self.query_collector.fetch_all(
            batch.extend_num_tokens if batch.forward_mode.is_extend() else len(batch.reqs)
        )
        if batch.forward_mode.is_extend():
            cunum_tokens = 0
            for req in batch.reqs:
                if req.rid not in req_to_remove:
                    entry: KVSelectEntry = self.entries.get(req.rid)
                    if entry is not None: # recompute, update the entry the same way as decode
                        seq_end_pos = cunum_tokens + req.extend_input_len
                        query = queries[:, seq_end_pos - 1:seq_end_pos]
                    else:
                        entry = KVSelectEntry(req.rid, req.origin_input_ids, req.output_ids)
                        query = queries[:, cunum_tokens:cunum_tokens + req.extend_input_len]
                        self.entries[req.rid] = entry
                    entry.cached_queries.append(query)
                cunum_tokens += req.extend_input_len
        else:
            for i, req in enumerate(batch.reqs):
                entry: KVSelectEntry = self.entries.get(req.rid)
                if entry is None or req.rid in req_to_remove:
                    continue # this request have already finished
                entry.cached_queries.append(queries[:, i:i + 1])

    def post_key_cache(self, rid: str, key_indices: torch.Tensor, write_num: int) -> None:
        entry: KVSelectEntry = self.entries.get(rid)
        if entry is None:
            logger.warning(f"Request {rid} not found in kv selector, ignoring posted key cache")
            return
        fetched_query_len = 0
        query_list = []
        while fetched_query_len < write_num:
            if len(entry.cached_queries) == 0:
                break
            query = entry.cached_queries.pop(0)
            fetched_query_len += query.shape[1]
            query_list.append(query)
        assert fetched_query_len == write_num, \
            f"{rid} Fetched query length {fetched_query_len} is not equal to write_num {write_num}"
        entry.accumu_length += write_num
        assert key_indices.shape[0] >= entry.accumu_length, \
            f"Key shape {key_indices.shape[0]} is less than accumulated length {entry.accumu_length}"
        query = torch.cat(query_list, dim=1)
        self.op_queue.put(KVSelectOperation(rid, query, key_indices[: entry.accumu_length].clone()))
        self.compute_op_count[rid] = self.compute_op_count.get(rid, 0) + write_num

    def restore_req(self, req: Req) -> None:
        if entry := self.entries.get(req.rid):
            entry.update_output_ids(req)
            entry.restore_req(req)
        else:
            logger.warning(f"Request {req.rid} not found in KVSelectEntry, ignoring restore")
    
    def select_ready(self, req: Req) -> bool:
        if req.rid in self.entries:
            entry: KVSelectEntry = self.entries[req.rid]
            if entry.topk_indices is None:
                return False
        return False

    def wait_for_ready(self, req: Req) -> bool:
        if req.rid in self.entries:
            entry: KVSelectEntry = self.entries[req.rid]
            while entry.compressed_indices is None and not self.stop_event.is_set():
                print(f"wait for kv selecting for req {req.rid}")
                time.sleep(0.5)
            return True
        else:
            logger.warning(f"Request {req.rid} not found in KVSelectEntry, ignoring wait")
        return False

    def req_finished(self, rid: str) -> None:
        if rid in self.compute_op_count:
            if self.compute_op_count[rid] > 0:
                self.finished_reqs.add(rid) # mark as finished
            else:
                del self.compute_op_count[rid]
        if entry := self.entries.get(rid):
            del self.entries[rid]
            if entry.compressed_indices is not None:
                self.host_free(entry.compressed_indices)
                entry.compressed_indices = None

    def _worker(self) -> None:
        torch.set_num_threads(NUM_TORCH_SUBPROCESSES)
        while not self.stop_event.is_set():
            try:
                op: KVSelectOperation = self.op_queue.get(timeout=1)
            except Empty:
                continue
            if op.rid not in self.finished_reqs and op.rid in self.entries:
                try:
                    # pass
                    self._compute_op(op)
                except Exception as e:
                    print(f"KVSelector: compute op {op.rid} failed: {e}")
                    print(f"KVSelector: op {op.rid} query shape {op.query.shape}, "
                          "key shape {op.key_indices.shape}")
                    raise e
            if op.rid in self.compute_op_count:
                self.compute_op_count[op.rid] -= 1
                if self.compute_op_count[op.rid] == 0:
                    if op.rid in self.finished_reqs:
                        del self.compute_op_count[op.rid]
                        self.finished_reqs.remove(op.rid)

    @torch.inference_mode()
    def _compute_op(self, op: KVSelectOperation) -> None:
        entry: KVSelectEntry = self.entries.get(op.rid)
        if entry is None:
            return
        # PHASE1: do self-attention computation on CPU
        query = op.query.view(self.layer_num, op.query.shape[1], -1, self.head_dim)
        key = torch.stack(
            [self.kv_buffer[0, layer, op.key_indices] for layer in SAMPLED_LAYERS], dim=0
        )
        key = key.repeat_interleave(query.shape[2] // key.shape[2], dim=2)
        time_start = time.time()
        scores = torch.einsum("lqhd,lkhd->lhqk", query, key) / (self.head_dim ** 0.5)
        scores = scores.to(dtype=torch.float32)
        mask = torch.tril(
            torch.ones((query.shape[1], key.shape[1]), dtype=torch.bool),
            diagonal=key.shape[1] - query.shape[1],
        )
        mask = mask.expand(self.layer_num, query.shape[2], -1, -1)
        scores = scores.masked_fill(mask.logical_not(), float("-inf"))
        scores = scores.softmax(dim=-1) # do softmax on the key length dimension
        scores = scores.sum(dim=(0, 1, 2)) # (key_len)
        # print(f"KVSelector: compute op {op.rid} scores {scores}")
        if entry.accumu_attn_scores is None:
            entry.accumu_attn_scores = scores # (key_len)
        else:
            scores[:entry.accumu_attn_scores.shape[0]] += entry.accumu_attn_scores
            entry.accumu_attn_scores = scores
        # print(f"KVSelector: compute op {op.rid} scores {entry.accumu_attn_scores}")

        # PHASE2: token merging
        compressed_len = len(entry.compressed_indices) if entry.compressed_indices is not None else 0
        scores = scores[compressed_len:]
        key_indices = op.key_indices[compressed_len:]
        scores = (scores / scores.sum()).tolist()
        if len(scores) < 512:
            return
        budget_len = max(int(len(scores) * entry.compress_ratio), 1)
        new_indices = self.host_alloc(budget_len)
        for layer_idx in range(self.kv_buffer.shape[1]):
            compressed_kv = [self.kv_buffer[:, layer_idx, idx].view(2, self.head_num * self.head_dim) 
                             for idx in key_indices]
            cur_scores = scores
            cos_similiarity_threshold = 0.9
            while len(compressed_kv) > budget_len and cos_similiarity_threshold >= 0.8:
                new_compressed_kv = [compressed_kv[0]]
                new_scores = [cur_scores[0]]
                assert len(compressed_kv) == len(cur_scores), \
                    f"KVSelector: compute op {op.rid} layer {layer_idx} compressed kv length {len(compressed_kv)} " \
                    f"and cur_scores length {len(cur_scores)} mismatch"
                for i in range(1, len(compressed_kv)):
                    similarity = cosine_similarity(
                        new_compressed_kv[-1][0], compressed_kv[i][0], dim=-1
                    ).item()
                    if similarity >= cos_similiarity_threshold:
                        ratio = new_scores[-1] / (new_scores[-1] + cur_scores[i])
                        new_scores[-1] += cur_scores[i]
                        new_compressed_kv[-1] = ratio * new_compressed_kv[-1] + \
                            (1 - ratio) * compressed_kv[i]
                    else:
                        new_compressed_kv.append(compressed_kv[i])
                        new_scores.append(cur_scores[i])
                # print(f"KVSelector: compute op {op.rid} layer {layer_idx} "
                #       f": {len(compressed_kv)} -> {len(new_compressed_kv)}, "
                #       f"threshold {cos_similiarity_threshold:.2f}", file=open("tmp/kv_selector.log", "a"))
                # update the compressed kv
                if len(new_compressed_kv) < budget_len:
                    break
                compressed_kv = new_compressed_kv
                cur_scores = new_scores
                cos_similiarity_threshold -= 0.03
            if len(compressed_kv) > budget_len:
                # select the top-k tokens
                topk_indices = sorted(range(len(compressed_kv)), key=lambda i: cur_scores[i], reverse=True)[:budget_len]
                compressed_kv = [compressed_kv[i] for i in topk_indices]
            if op.rid not in self.entries:
                self.host_free(new_indices)
                logger.warning(f"Request {op.rid} not found in KVSelectEntry, ignoring compute op")
                return
            self.kv_buffer[:, layer_idx, new_indices] = torch.stack(
                compressed_kv, dim=1).view(2, budget_len, self.head_num, self.head_dim)
        if entry.compressed_indices is not None:
            new_indices = torch.cat([entry.compressed_indices, new_indices], dim=0)
        entry.compressed_indices = new_indices

        elapsed_time = time.time() - time_start
        entry.total_compute_time += elapsed_time
        entry.last_compute_time = elapsed_time
        # print(f"KVSelector: compute op {op.rid} time {elapsed_time:.4f}s, "
        #       f"accumulated time {entry.total_compute_time:.4f}s")
