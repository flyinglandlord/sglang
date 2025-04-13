import torch
import logging
import threading
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Optional
from queue import Empty, Full, Queue

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ForwardMode
from sglang.srt.selective_loading.query_collector import QueryCollector

logger = logging.getLogger(__name__)

NUM_TORCH_SUBPROCESSES = 32


@dataclass
class KVSelectOperation:
    rid: str
    query: torch.Tensor
    key: torch.Tensor


class KVSelectEntry:
    def __init__(self, rid: str, input_ids: Tuple[int], output_ids: List[int]):
        self.rid = rid
        self.input_ids = input_ids
        self.output_ids = output_ids
        # NOTE: `last_length` is the length of the request last seen by this entry
        # it is used to update output_ids when new tokens are generated.
        # Also, when the request undergoes a kv selection, the `last_length`
        # is the length of the selected input_ids + output_ids.
        self.last_length = len(input_ids) + len(output_ids)
    
    def update_output_ids(self, req: Req):
        assert req.rid == self.rid, f"Request ID mismatch: {req.rid} != {self.rid}"
        num_token_new = len(req.origin_input_ids) + len(req.output_ids) - self.last_length
        assert len(req.output_ids) >= num_token_new >= 0, \
            f"Invalid token update: {num_token_new} new tokens, " \
            f"output_ids length {len(req.output_ids)}, last_length {self.last_length}"
        self.output_ids.extend(req.output_ids[-num_token_new:])
        self.last_length += num_token_new
    
    def restore_req(self, req: Req):
        assert req.rid == self.rid, f"Request ID mismatch: {req.rid} != {self.rid}"
        req.input_ids = self.input_ids
        req.output_ids = self.output_ids
        self.last_length = len(self.input_ids) + len(self.output_ids)


class KVSelector:
    def __init__(self, head_num: int, head_dim: int, layer_num: int):
        self.entries: Dict[str, KVSelectEntry] = {}
        self.compute_op_count: Dict[str, int] = {}
        self.finished_reqs: Set[str] = set()
        self.cached_queries: Dict[str, Queue] = {}
        self.query_collector = QueryCollector()
        self.op_queue = Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        print(f"KVSelector: head_num {head_num}, head_dim {head_dim}, layer_num {layer_num}")
    
    def reset(self):
        self.stop_event.set()
        self.worker.join()
        self.entries.clear()
        self.compute_op_count.clear()
        self.finished_reqs.clear()
        self.cached_queries.clear()
        self.op_queue = Queue()
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def update_with_batch(self, batch: ScheduleBatch) -> None:
        """NOTE: The query tensors captured by QueryCollector are not associated
        with any request ID. This function matches the query tensors with the
        last batch of requests in order."""
        queries = self.query_collector.fetch_all()
        if batch.forward_mode.is_extend():
            assert queries.shape[0] == batch.extend_num_tokens, \
                f"Query shape {queries.shape[0]} does not match extend_num_tokens {batch.extend_num_tokens}"
            cunum_tokens = 0
            for req in batch.reqs:
                assert req.rid not in self.entries, \
                    f"Prefill request {req.rid} already exists in KVSelectEntry"
                self.entries[req.rid] = KVSelectEntry(req.rid, req.origin_input_ids, req.output_ids)
                self.cached_queries[req.rid] = Queue()
                query = queries[cunum_tokens:cunum_tokens + req.extend_input_len]
                self.cached_queries[req.rid].put(query)
                cunum_tokens += req.extend_input_len
        else:
            assert queries.shape[0] == len(batch.reqs), \
                f"Query shape {queries.shape[0]} does not match num_tokens {batch.num_tokens}"
            for i, req in enumerate(batch.reqs):
                assert req.rid in self.entries, \
                    f"Request {req.rid} not found in KVSelectEntry"
                assert req.rid in self.cached_queries, \
                    f"Request {req.rid} not found in cached queries"
                self.entries[req.rid].update_output_ids(req)
                query = queries[i:i + 1]
                self.cached_queries[req.rid].put(query)

    def post_key_cache(self, rid: str, key: torch.Tensor) -> None:
        if rid not in self.cached_queries:
            logger.warning(f"Request {rid} not found in cached queries, ignoring posted key cache")
            return
        try:
            query = self.cached_queries[rid].get_nowait()
        except Empty:
            logger.warning(f"Request {rid} has no cached query, ignoring posted key cache")
            return
        self.op_queue.put(KVSelectOperation(rid, query, key))
        self.compute_op_count[rid] = self.compute_op_count.get(rid, 0) + 1

    def restore_req(self, req: Req) -> None:
        if req.rid in self.entries:
            self.entries[req.rid].restore_req(req)
        else:
            logger.warning(f"Request {req.rid} not found in KVSelectEntry, ignoring restore")
    
    def req_finished(self, rid: str) -> None:
        if rid in self.compute_op_count:
            if self.compute_op_count[rid] > 0:
                self.finished_reqs.add(rid) # mark as finished
            else:
                del self.compute_op_count[rid]
        if rid in self.cached_queries:
            del self.cached_queries[rid]
        if rid in self.entries:
            del self.entries[rid]

    def _worker(self) -> None:
        torch.set_num_threads(NUM_TORCH_SUBPROCESSES)
        while not self.stop_event.is_set():
            try:
                op = self.op_queue.get(timeout=1)
            except Empty:
                continue
            if op.rid in self.finished_reqs:
                logger.warning(f"Request {op.rid} is already finished, skipping KV selection")
                continue
            if op.rid not in self.entries:
                logger.warning(f"Request {op.rid} not found in KVSelectEntry, skipping KV selection")
                continue
            self._compute_op(op)
            if op.rid in self.compute_op_count:
                self.compute_op_count[op.rid] -= 1
                if self.compute_op_count[op.rid] == 0:
                    del self.compute_op_count[op.rid]
                    self.finished_reqs.remove(op.rid)
    
    @torch.inference_mode()
    def _compute_op(self, op: KVSelectOperation) -> None:
        print(f"KVSelector: computing op for {op.rid}, query shape {op.query.shape}, key shape {op.key.shape}")
        # TODO: This is a placeholder for the actual computation
        import time
        time.sleep(0.1)
