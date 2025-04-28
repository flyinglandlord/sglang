import time
import torch
import logging
import threading
from multiprocessing import Pool
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Optional
from queue import Empty, Full, Queue

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.selective_loading.query_collector import QueryCollector, SAMPLED_LAYERS

logger = logging.getLogger(__name__)

NUM_TORCH_SUBPROCESSES = 32


@dataclass
class KVSelectOperation:
    rid: str
    query: torch.Tensor
    key_indices: torch.Tensor


class KVSelectEntry:
    def __init__(self, rid: str, input_ids: Tuple[int], output_ids: List[int]):
        self.rid = rid
        self.fill_ids: List[int] = input_ids + output_ids
        self.originial_input_len: int = len(input_ids)
        self.accumu_length: int = 0
        self.last_compute_time: float = 0.0
        self.total_compute_time: float = 0.0
        self.accumu_attn_scores: Optional[torch.Tensor] = None
        self.last_length: int = len(self.fill_ids)
        self.cached_queries: List[torch.Tensor] = []
    
    def update_output_ids(self, req: Req):
        assert req.rid == self.rid, f"Request ID mismatch: {req.rid} != {self.rid}"
        num_token_new = len(req.origin_input_ids) + len(req.output_ids) - self.last_length
        assert len(req.output_ids) >= num_token_new >= 0, \
            f"Invalid token update: {num_token_new} new tokens, " \
            f"output_ids length {len(req.output_ids)}, last_length {self.last_length}"
        self.fill_ids.extend(req.output_ids[-num_token_new:])
        self.last_length += num_token_new
    
    def restore_req(self, req: Req):
        assert req.rid == self.rid, f"Request ID mismatch: {req.rid} != {self.rid}"
        req.origin_input_ids = self.fill_ids[:self.originial_input_len]
        req.output_ids = self.fill_ids[self.originial_input_len:]
        self.last_length = len(self.fill_ids)


class KVSelector:
    def __init__(self, head_num: int, head_dim: int, k_buffer: torch.Tensor):
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
        self.k_buffer = k_buffer 
    
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
                    if entry is not None:
                        entry.update_output_ids(req)
                        seq_end_pos = cunum_tokens + req.extend_input_len
                        query = queries[:, cunum_tokens:seq_end_pos]
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
                entry.update_output_ids(req)
                entry.cached_queries.append(queries[:, i:i + 1])

    def post_key_cache(self, rid: str, key_indices: torch.Tensor) -> None:
        entry: KVSelectEntry = self.entries.get(rid)
        if entry is None:
            logger.warning(f"Request {rid} not found in kv selector, ignoring posted key cache")
            return
        if len(entry.cached_queries) == 0:
            logger.warning(f"Request {rid} has no cached query, ignoring posted key cache")
            return
        query = entry.cached_queries.pop(0)
        entry.accumu_length += query.shape[1]
        assert key_indices.shape[1] >= entry.accumu_length, \
            f"Key shape {key_indices.shape[1]} is less than accumulated length {entry.accumu_length}"
        self.op_queue.put(KVSelectOperation(rid, query, key_indices[: entry.accumu_length]))
        self.compute_op_count[rid] = self.compute_op_count.get(rid, 0) + 1

    def restore_req(self, req: Req) -> None:
        if req.rid in self.entries:
            self.entries[req.rid].restore_req(req)
        else:
            logger.warning(f"Request {req.rid} not found in KVSelectEntry, ignoring restore")
    
    def select_ready(self, req: Req) -> bool:
        if req.rid in self.entries:
            entry: KVSelectEntry = self.entries[req.rid]
            if entry.accumu_attn_scores is None:
                return False
        return False

    def wait_for_ready(self, req: Req) -> bool:
        if req.rid in self.entries:
            entry: KVSelectEntry = self.entries[req.rid]
            while entry.accumu_attn_scores is None and not self.stop_event.is_set():
                time.sleep(0.01)
            return True
        else:
            logger.warning(f"Request {req.rid} not found in KVSelectEntry, ignoring wait")
        return False
    
    def select_kv(self, req: Req, target_length: int) -> Optional[torch.Tensor]:
        entry: KVSelectEntry = self.entries.get(req.rid)
        if entry is None or entry.accumu_attn_scores is None:
            return None
        attn_scores = entry.accumu_attn_scores
        full_length = len(entry.fill_ids) - 1
        print(f'req {req.rid} attn_scores shape {attn_scores.shape}, full_length {full_length}')
        keep_length = max(64, full_length - attn_scores.shape[0])
        if keep_length > target_length:
            return None
        _, indices = attn_scores.topk(target_length - keep_length, dim=0, largest=True, sorted=False)
        indices = torch.cat([indices, torch.arange(full_length - keep_length, full_length)])
        req.origin_input_ids = []
        req.output_ids = []
        for i in indices:
            if i < entry.originial_input_len:
                req.origin_input_ids.append(entry.fill_ids[i])
            else:
                req.output_ids.append(entry.fill_ids[i])
        entry.last_length = indices.shape[0]
        return indices

    def req_finished(self, rid: str) -> None:
        if rid in self.compute_op_count:
            if self.compute_op_count[rid] > 0:
                self.finished_reqs.add(rid) # mark as finished
            else:
                del self.compute_op_count[rid]
        if rid in self.entries:
            del self.entries[rid]

    def _worker(self) -> None:
        torch.set_num_threads(NUM_TORCH_SUBPROCESSES)
        while not self.stop_event.is_set():
            try:
                op: KVSelectOperation = self.op_queue.get(timeout=1)
            except Empty:
                continue
            if op.rid not in self.finished_reqs and op.rid in self.entries:
                try:
                    pass
                    # self._compute_op(op)
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
        # do self-attention computation on CPU
        query = op.query.view(self.layer_num, op.query.shape[1], -1, self.head_dim)
        query = query.transpose(1, 2) # (layer_num, head_num, seq_length, head_dim)
        key = op.key.transpose(1, 2) # (layer_num, head_num, seq_length, head_dim)
        # print(f"KVSelector: compute op {op.rid} query shape {query.shape}, key shape {key.shape}")
        time_start = time.time()
        group_query_num = query.shape[1] // key.shape[1]
        # repeat key
        key = key.repeat_interleave(group_query_num, dim=1)
        # compute attention scores
        scores = torch.einsum("lhqd,lhkd->lhqk", query, key) / (self.head_dim ** 0.5)
        scores = scores.softmax(dim=3) # do softmax on the key length dimension
        scores = scores.sum(dim=(0, 1, 2)) # (key_len)
        if entry.accumu_attn_scores is None:
            entry.accumu_attn_scores = scores # (key_len)
        else:
            scores[:-1] += entry.accumu_attn_scores
            entry.accumu_attn_scores = scores
        elapsed_time = time.time() - time_start
        entry.total_compute_time += elapsed_time
        entry.last_compute_time = elapsed_time
        # print(f"KVSelector: compute op {op.rid} time {elapsed_time:.4f}s, "
        #       f"accumulated time {entry.total_compute_time:.4f}s")
