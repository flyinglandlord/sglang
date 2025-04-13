from __future__ import annotations
import time
import torch
import threading

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Set, Dict
from queue import Empty

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from python.sglang.srt.mem_cache.chunk_cache import ChunkCache, ChunkCacheEntry
from sglang.srt.managers.cache_controller import HiCacheController, CacheOperation
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.memory_pool import (
    BaseTokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPoolHost,
    MemoryStateInt,
    ReqToTokenPool,
)
from sglang.srt.selective_loading.kv_selector import KVSelector
from sglang.srt.selective_loading.query_collector import ENABLE_QUERY_COLLECTOR

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class SyncChunkCache(ChunkCache):
    def __init__(
        self, 
        req_to_token_pool: ReqToTokenPool, 
        token_to_kv_pool: BaseTokenToKVPool,
    ):
        self.req_write_op_count = {}
        self.req_to_remove: Dict[str, Optional[List[int]]] = {}
        self.req_to_evict: Dict[str, int] = {}
        self.kv_selector: Optional[KVSelector] = None
        self.token_to_kv_pool_host = MLATokenToKVPoolHost(token_to_kv_pool)
        self.cache_controller = HiCacheController(
            token_to_kv_pool, self.token_to_kv_pool_host, self.req_to_remove
        )
        self.stop_event = threading.Event()
        self.entries_lock = threading.Lock()
        self.poller = threading.Thread(target=self._poll_write_ack_queue, daemon=True)
        super().__init__(req_to_token_pool, token_to_kv_pool)

    def init_kv_selector(self) -> Optional[KVSelector]:
        """Initialize the KV selector if not already initialized.
        Return None if the KV selector is not used."""
        if (
            ENABLE_QUERY_COLLECTOR
            and self.kv_selector is None
            and isinstance(self.token_to_kv_pool, MHATokenToKVPool)
        ):
            self.kv_selector = KVSelector(
                self.token_to_kv_pool.head_num,
                self.token_to_kv_pool.head_dim,
                self.token_to_kv_pool.layer_num,
            )
        return self.kv_selector

    def reset(self):
        self.stop_event.set()
        if self.poller.is_alive():
            self.poller.join(timeout=2)
        self.stop_event.clear()
        if self.kv_selector:
            self.kv_selector.reset()
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.req_write_op_count.clear()
        self.req_to_remove.clear()
        self.req_to_evict.clear()
        super().reset()
        self.poller = threading.Thread(target=self._poll_write_ack_queue, daemon=True)
        self.poller.start()

    def _cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        # free host memory
        if req.rid in self.entries:
            entry = self.entries[req.rid]
            if entry.host_value is not None:
                self.token_to_kv_pool_host.free(entry.host_value)
                entry.host_value = None
                entry.host_req_pool_idx = None
        if req.rid in self.req_write_op_count:
            del self.req_write_op_count[req.rid]
        if self.kv_selector:
            self.kv_selector.req_finished(req.rid)
        # then call base class to free device memory
        super().cache_finished_req(req, token_ids)

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        pass # override to do nothing, the functionality is moved to sync_prefill and sync_decode

    def remove_req(self, req: Req):
        assert False, "remove_req is not supported in SyncChunkCache"

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        # NOTE: We must use this asynchronized way to remove finished requests
        # since we write kv cache to cpu memory in a separate thread.
        # The cache will finally be remove when polling the write ack queue.
        with self.entries_lock:
            if self.req_write_op_count.get(req.rid, 0) > 0:
                self.req_to_remove[req.rid] = token_ids
            else: # remove the request immediately
                self._cache_finished_req(req, token_ids)

    def evict_device(self, req: Req, seq_len: int):
        # NOTE: Also asynchronized way to evict device memory
        with self.entries_lock:
            if self.req_write_op_count.get(req.rid, 0) > 0:
                self.req_to_evict[req.rid] = seq_len
            else: # evict the request immediately
                self._evict_device(req, seq_len)

    def get_evicting_reqs(self) -> List[str]:
        return list(self.req_to_evict.keys())

    def _evict_device(self, req: Req, seq_len: int):
        # evict a request in device memory, but still exist in host memory
        # first check if the request is in cache and no IO operation is ongoing
        if req.rid not in self.entries:
            raise RuntimeError(f"Request {req.rid} not in cache")
        if req.rid in self.req_write_op_count and self.req_write_op_count[req.rid] > 0:
            raise RuntimeError(f"Request {req.rid} is writing")
        if req.last_node.loading:
            raise RuntimeError(f"Request {req.rid} is loading")
        # evict the device memory
        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : seq_len
        ]
        entry = self.entries[req.rid]
        if not self.token_to_kv_pool_host.is_synced(entry.host_value):
            raise RuntimeError(f"Request {req.rid} is not synced")
        self.token_to_kv_pool_host.update_backup(entry.host_value)
        # print('device indices: ', device_indices.detach().cpu().numpy())
        # print('Available device pool size after evict: ', self.token_to_kv_pool.available_size())
        self.token_to_kv_pool.free(device_indices)
        self.req_to_token_pool.free(req.req_pool_idx)
        entry.value = None
        entry.evicted = True
        req.last_node = entry
        req.req_pool_idx = None

    def load_back(self, req: Req):
        rid = req.rid
        if rid not in self.entries:
            raise RuntimeError(f"Request {rid} not in cache")
        entry = self.entries[rid]
        if not entry.backuped:
            raise RuntimeError(f"Request {rid} not backuped")
        if entry.host_value is None:
            raise RuntimeError(f"Host value is None for rid {rid}")
        if entry.loading:
            raise RuntimeError(f"Request {rid} is loading")
        if rid in self.req_write_op_count and self.req_write_op_count[rid] > 0:
            raise RuntimeError(f"Request {rid} is writing")
        if self.token_to_kv_pool_host.get_state(entry.host_value) != MemoryStateInt.BACKUP:
            raise RuntimeError(f"Request {rid} is not in host memory")
        # allocate device memory
        device_indices = self.cache_controller.load(
            entry.host_value, node_id=entry
        )
        if device_indices is None:
            raise RuntimeError(f'Failed to allocate device memory for request {rid}')
        # update the entry
        req.req_pool_idx = self.req_to_token_pool.alloc(1)[0]
        entry.loading = True
        entry.value = device_indices
        entry.evicted = False

    def load_check(self, req: Optional[Req] = None) -> Optional[bool]:
        # synchronize the loading status
        while not self.cache_controller.ack_load_queue.empty():
            try:
                ack = self.cache_controller.ack_load_queue.get_nowait()
                entry = self.entries[ack.rid]
                if entry is None:
                    raise RuntimeError(f"Failed to find rid {ack.rid} in entry")
                if not entry.loading:
                    raise RuntimeError(f"Entry {ack.rid} is not loading")
                entry.loading = False
            except Exception as e:
                break
        if req is not None and req.rid in self.entries:
            return not self.entries[req.rid].loading
        
    def _poll_write_ack_queue(self):
        while not self.stop_event.is_set():
            try:
                ack = self.cache_controller.ack_write_queue.get(timeout=1)
                entry = self.entries[ack.rid]
                if entry is None:
                    raise RuntimeError(f"Failed to find rid {ack.rid} in entry")
                if self.kv_selector:
                    k_cache = self.token_to_kv_pool_host.get_flat_data(
                        entry.host_value
                    )[0]
                    self.kv_selector.post_key_cache(ack.rid, k_cache)
                with self.entries_lock:
                    self.req_write_op_count[ack.rid] -= 1
                    if self.req_write_op_count[ack.rid] == 0:
                        # remove the request from cache
                        if ack.rid in self.req_to_evict:
                            seq_len = self.req_to_evict[ack.rid]
                            del self.req_to_evict[ack.rid]
                            self._evict_device(ack.req, seq_len)
                        elif ack.rid in self.req_to_remove:
                            token_ids = self.req_to_remove[ack.rid]
                            del self.req_to_remove[ack.rid]
                            self._cache_finished_req(ack.req, token_ids)
            except Empty:
                continue
            except Exception as e:
                raise e

    def can_load_back(self, req: Req) -> bool:
        # check if the request can be loaded back
        if req.rid not in self.entries:
            return False
        entry = self.entries[req.rid]
        if entry.loading:
            raise RuntimeError(f"Request {req.rid} is loading")
        return entry.host_value is not None

    def wait_write(self, req: Req):
        # [deprecated] wait for a certain request to finish writing
        if req.rid not in self.entries:
            raise RuntimeError(f"Request {req.rid} not in cache")
        if req.rid not in self.req_write_op_count:
            raise RuntimeError(f"Request {req.rid} not in write op count")
        retry = 0
        while self.req_write_op_count[req.rid] > 0:
            time.sleep(1e-4)
            retry += 1
        # assert retry < 100000, f"Wait write op for {req.rid} timeout, count: {self.req_write_op_count}"

    def sync_decode(self, req: Req, seq_len: int):
        # add a write operation for this one token generated in this step
        if req.rid not in self.entries:
            # print(f"WARNING: Request {req.rid} not in cache")
            return # scheduler will remove request before the my_scheduler call this
        entry: ChunkCacheEntry = self.entries[req.rid]
        if entry.host_value is None:
            raise RuntimeError(f"Host value is None for rid {req.rid}")
        host_indices = self.token_to_kv_pool_host.alloc(1)
        if host_indices is None:
            raise RuntimeError("Failed to allocate host memory for backup")
        # original cache_controller.write writes the whole request
        # so we need to modify it to write only the last token
        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, seq_len - 1: seq_len
        ]
        self.token_to_kv_pool_host.protect_write(host_indices)
        self.cache_controller.write_queue.put(
            CacheOperation(host_indices, device_indices, entry)
        )
        entry.host_value = torch.cat([entry.host_value, host_indices], dim=0)
        entry.value = torch.cat([entry.value, device_indices], dim=0)
        entry.backuped = True
        self.req_write_op_count[req.rid] += 1
    
    def sync_prefill(self, req: Req, seq_len: int):
        # add a write operation for this one token generated in this step
        if req.rid in self.entries:
            raise RuntimeError(f"Request {req.rid} already in cache")
        if req.rid in self.req_write_op_count and self.req_write_op_count[req.rid] > 0:
            raise RuntimeError(f"Request {req.rid} already in write op count")
        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : seq_len
        ]
        entry = ChunkCacheEntry(req.rid, device_indices)
        self.entries[req.rid] = entry
        entry.value = device_indices
        entry.req = req
        req.last_node = entry
        host_indices = self.cache_controller.write(
            device_indices=device_indices, node_id=entry
        )
        if host_indices is None:
            raise RuntimeError("Failed to allocate host memory for backup")
        entry.host_value = host_indices
        entry.backuped = True
        self.req_write_op_count[req.rid] = 1

    def sync_batch(self, batch: ScheduleBatch):
        seq_lens_cpu = batch.seq_lens.cpu()
        if self.kv_selector:
            self.kv_selector.update_with_batch(batch)
        with self.entries_lock:
            for i, req in enumerate(batch.reqs):
                if batch.forward_mode.is_extend():
                    self.sync_prefill(req, seq_lens_cpu[i])
                else:
                    self.sync_decode(req, seq_lens_cpu[i])
