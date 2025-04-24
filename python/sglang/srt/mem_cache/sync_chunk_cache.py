from __future__ import annotations
import time
import torch
import threading

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Set, Dict
from queue import Empty, Queue
from dataclasses import dataclass

from python.sglang.srt.mem_cache.chunk_cache import ChunkCache, ChunkCacheEntry
from sglang.srt.managers.cache_controller import HiCacheController, CacheOperation
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPoolHost,
    ReqToTokenPool,
)
from sglang.srt.selective_loading.kv_selector import KVSelector
from sglang.srt.selective_loading.query_collector import ENABLE_QUERY_COLLECTOR

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class SyncCacheEntry(ChunkCacheEntry):
    def __init__(self, rid: str, value: torch.Tensor):
        super().__init__(rid, value)
        self.last_writing_pos: int = 0
        self.evicted_len: int = 0
        self.written_len: int = 0
        self.is_synced: bool = True

class SyncChunkCache(ChunkCache):
    def __init__(
        self, 
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: MHATokenToKVPool,
        sync_chunk_size: int = 64,
    ):
        assert isinstance(token_to_kv_pool, MHATokenToKVPool), \
            f"token_to_kv_pool must be MHATokenToKVPool, but got {type(req_to_token_pool)}"
        self.req_to_remove: Dict[str, Optional[List[int]]] = {}
        self.req_to_evict: Dict[str, int] = {}
        self.kv_selector: Optional[KVSelector] = None
        self.token_to_kv_pool = token_to_kv_pool
        self.token_to_kv_pool_host = MLATokenToKVPoolHost(token_to_kv_pool)
        # data fields for loading and writing profilers
        initial_load_speed, self.write_speed = self._initial_profile()
        print(f"Initial load speed: {initial_load_speed:.2f} tokens/s, write speed: {self.write_speed:.2f} tokens/s")
        self.loading_token_num = 0
        self.write_token_num = 0
        self.wrote_token_num = 0
        self.writing_records: Dict[str, Queue[int]] = {}
        self.cache_controller = HiCacheController(
            token_to_kv_pool, self.token_to_kv_pool_host, initial_load_speed, self.req_to_remove,
        )
        self.stop_event = threading.Event()
        self.entries_lock = threading.Lock()
        self.poller = threading.Thread(target=self._poll_write_ack_queue, daemon=True)
        self.sync_chunk_size = sync_chunk_size
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
        if self.kv_selector is not None:
            self.kv_selector.reset()
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.req_to_remove.clear()
        self.req_to_evict.clear()
        self.writing_records.clear()
        self.loading_token_num = 0
        self.write_token_num = 0
        self.wrote_token_num = 0
        super().reset()
        self.poller = threading.Thread(target=self._poll_write_ack_queue, daemon=True)
        self.poller.start()

    def _cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        # free host memory
        if req.rid in self.entries:
            entry: SyncCacheEntry = self.entries[req.rid]
            if entry.host_value is not None:
                self.token_to_kv_pool_host.free(entry.host_value)
                entry.host_value = None
                entry.host_req_pool_idx = None
        else: 
            print(f"WARNING: Request {req.rid} finished right after prefill")
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
            record = self.writing_records.get(req.rid)
            if record is None or record.empty():
                self._cache_finished_req(req, token_ids)
                if record is not None:
                    del self.writing_records[req.rid]
            self.req_to_remove[req.rid] = token_ids
        if self.kv_selector is not None:
            self.kv_selector.req_finished(req.rid)

    def evict_device(self, req: Req, seq_len: int):
        # NOTE: Also asynchronized way to evict device memory
        assert req.rid in self.entries, f"Request {req.rid} not in cache"
        with self.entries_lock:
            self._write_host(self.entries[req.rid], req, True)
            if req.rid in self.writing_records and not self.writing_records[req.rid].empty():
                self.req_to_evict[req.rid] = seq_len
            else: # evict the request immediately
                self._evict_device(req, seq_len)

    def get_evicting_reqs(self) -> List[str]:
        return list(self.req_to_evict.keys())
    
    def get_removing_reqs(self) -> List[str]:
        return list(self.req_to_remove.keys())

    def _evict_device(self, req: Req, evict_len: int):
        # evict a request in device memory, but still exist in host memory
        if req.rid not in self.entries:
            raise RuntimeError(f"Request {req.rid} not in cache")
        if req.last_node is not None and req.last_node.loading:
            raise RuntimeError(f"Request {req.rid} is loading")
        # evict the device memory
        entry: SyncCacheEntry = self.entries[req.rid]
        assert entry.value is not None, f"Request {req.rid} value is None"
        # print(f"Evicting {evict_len} tokens from request {req.rid}, value shape: {entry.value.shape}")
        if evict_len < entry.value.shape[0]: # evict part of the device memory
            print(f"partially evicting {evict_len} tokens from request {req.rid}, value shape: {entry.value.shape}")
            value_evict = entry.value[:evict_len]
            self.token_to_kv_pool.free(value_evict)
            entry.value = entry.value[evict_len:]
            return
        if not self.writing_records[req.rid].empty():
            raise RuntimeError(f"Request {req.rid} is writing")
        # evict the whole device memory, but still keep the host memory
        # print(f"Evicting whole device memory for request {req.rid}, value shape: {entry.value.shape}")
        if entry.is_synced:
            self.token_to_kv_pool_host.update_backup(entry.host_value)
        self.token_to_kv_pool.free(entry.value)
        self.req_to_token_pool.free(req.req_pool_idx)
        entry.value = None
        entry.evicted = True
        req.last_node = entry
        req.req_pool_idx = None

    def load_back(self, req: Req, load_indices: Optional[torch.Tensor] = None):
        rid = req.rid
        with self.entries_lock:
            if rid not in self.entries:
                raise RuntimeError(f"Request {rid} not in cache")
            entry: SyncCacheEntry = self.entries[rid]
            if not entry.backuped:
                raise RuntimeError(f"Request {rid} not backuped")
            if entry.host_value is None:
                raise RuntimeError(f"Host value is None for rid {rid}")
            if entry.loading:
                raise RuntimeError(f"Request {rid} is loading")
            if not self.writing_records[req.rid].empty():
                raise RuntimeError(f"Request {rid} is writing")
        # allocate device memory
        host_value = entry.host_value
        if load_indices is not None:
            host_value = host_value[load_indices]
            assert host_value.shape[0] == load_indices.shape[0], \
                f"host_value {host_value.shape[0]} != load_indices {load_indices.shape[0]}"
        device_indices = self.cache_controller.load(host_value, node_id=entry)
        if device_indices is None:
            raise RuntimeError(f'Failed to allocate device memory for request {rid}')
        # update the entry
        req.req_pool_idx = self.req_to_token_pool.alloc(1)[0]
        entry.loading = True
        entry.value = device_indices
        entry.evicted = False
        entry.evicted_len = 0
        self.loading_token_num += device_indices.shape[0]

    def get_loading_workload(self) -> Tuple[int, float]:
        self.load_check()
        return self.loading_token_num, self.cache_controller.running_load_speed
    
    def get_writing_workload(self, rid: Optional[str] = None) -> Tuple[int, float]:
        if rid is None:
            return self.write_token_num - self.wrote_token_num, self.write_speed
        if rid not in self.entries:
            raise RuntimeError(f"Request {rid} not in cache")
        entry: SyncCacheEntry = self.entries[rid]
        return max(entry.last_writing_pos - self.wrote_token_num, 0), self.write_speed
    
    def select_and_load_back(self, req: Req, target_length: int, wait: bool = False):
        if self.kv_selector is None:
            return self.load_back(req)
        assert req.rid in self.entries, f"Request {req.rid} not in cache"
        if target_length > self.entries[req.rid].host_value.shape[0]:
            print(f"target_length {target_length} less than host_vlaue length {self.entries[req.rid].host_value.shape[0]} rid: {req.rid}")
            return self.load_back(req)
        if wait:
            self.kv_selector.wait_for_ready(req)
        indices = self.kv_selector.select_kv(req, target_length)
        print(f"Request {req.rid} selected kv {indices.shape[0]} for loading")
        if indices is None:
            raise RuntimeError(f"Failed to select kv for request {req.rid}")
        self.load_back(req, indices)

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
                self.loading_token_num -= ack.value.shape[0]
            except Exception as e:
                break
        if req is not None and req.rid in self.entries:
            return not self.entries[req.rid].loading
        
    def _poll_write_ack_queue(self):
        while not self.stop_event.is_set():
            try:
                ack = self.cache_controller.ack_write_queue.get(timeout=1)
                with self.entries_lock:
                    entry: SyncCacheEntry = self.entries.get(ack.rid)
                    assert ack.rid in self.writing_records, \
                        f"Request {ack.rid} not in writing records"
                    record = self.writing_records[ack.rid]
                    write_num = record.get_nowait()
                    self.wrote_token_num += write_num
                    if entry is not None:
                        if ack.rid in self.req_to_remove:
                            if not self.cache_controller.is_writing(ack.rid):
                                token_ids = self.req_to_remove[ack.rid]
                                self._cache_finished_req(ack.req, token_ids)
                        else:
                            entry.written_len += write_num
                            if entry.rid in self.req_to_evict:
                                self._evict_device(ack.req, entry.written_len - entry.evicted_len)
                                entry.evicted_len = entry.written_len
                            if self.kv_selector is not None:
                                k_cache = self.token_to_kv_pool_host.get_flat_data(
                                    entry.host_value
                                )[0]
                                self.kv_selector.post_key_cache(ack.rid, k_cache)
                    if record.empty():
                        if ack.rid in self.req_to_evict:
                            del self.req_to_evict[ack.rid]
                        elif ack.rid in self.req_to_remove:
                            del self.req_to_remove[ack.rid]
                            del self.writing_records[ack.rid]
            except Empty:
                continue
            except Exception as e:
                raise e

    def can_load_back(self, req: Req) -> bool:
        # check if the request can be loaded back
        if req.rid not in self.entries:
            return False
        entry: SyncCacheEntry = self.entries[req.rid]
        if entry.loading:
            raise RuntimeError(f"Request {req.rid} is loading")
        return entry.host_value is not None and entry.is_synced

    def wait_write(self, req: Req):
        # [deprecated] wait for a certain request to finish writing
        if req.rid not in self.entries:
            raise RuntimeError(f"Request {req.rid} not in cache")
        if req.rid not in self.writing_records:
            raise RuntimeError(f"Request {req.rid} not in write op count")
        record = self.writing_records[req.rid]
        start_time = time.time()
        while not record.empty():
            time.sleep(1e-4)
            if time.time() - start_time > 5:
                print(self.writing_records)
                raise RuntimeError(f"Request {req.rid} write op timeout")
    
    def _write_host(self, entry: SyncCacheEntry, req: Req, force_write: bool = False):
        is_prefill = entry.host_value is None
        if is_prefill:
            device_indices = entry.value
            sync_len = int(device_indices.shape[0])
        else:
            host_len = entry.host_value.shape[0]
            sync_len = entry.value.shape[0] - host_len
            if sync_len == 0:
                return # no need to write
            if not force_write and sync_len < self.sync_chunk_size:
                return # wait for more tokens to write
            device_indices = entry.value[host_len: host_len + sync_len]
        host_indices = self.cache_controller.write(device_indices, node_id=entry)
        if host_indices is None:
            raise RuntimeError("Failed to allocate host memory for backup")
        if is_prefill:
            assert entry.host_value is None, \
                f"Request {req.rid} host value is not None, {entry.host_value.shape}"
        else:
            host_indices = torch.cat([entry.host_value, host_indices], dim=0)
        entry.host_value = host_indices
        self.write_token_num += sync_len
        entry.backuped = True
        entry.last_writing_pos = self.write_token_num
        if is_prefill:
            record = Queue()
            record.put(sync_len)
            self.writing_records[req.rid] = record
        else:
            self.writing_records[req.rid].put(sync_len)

    def _sync_decode(self, req: Req, seq_len: int, is_recompute: bool = False):
        # add a write operation for this one token generated in this step
        if req.rid not in self.entries:
            # print(f"WARNING: Request {req.rid} not in cache")
            return # scheduler will remove request before the my_scheduler call this
        entry: SyncCacheEntry = self.entries[req.rid]
        if is_recompute and not entry.evicted:
            raise RuntimeError(f"Request {req.rid} not evicted, recompute is not allowed")
        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : seq_len
        ]
        entry.value = device_indices
        if is_recompute:
            entry.evicted = False
            entry.evicted_len = 0
            if entry.host_value is not None:
                self.token_to_kv_pool_host.update_synced(entry.host_value)
        if entry.is_synced:
            self._write_host(entry, req)
    
    def _sync_prefill(self, req: Req, seq_len: int):
        # add a write operation for this one token generated in this step
        if req.rid in self.entries:
            # recomputing a request is the same as decoding
            return self._sync_decode(req, seq_len, True)
        if req.rid in self.writing_records and not self.writing_records[req.rid].empty():
            raise RuntimeError(f"Request {req.rid} already in write op count")
        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : seq_len
        ]
        entry = SyncCacheEntry(req.rid, device_indices)
        self.entries[req.rid] = entry
        entry.req = req
        req.last_node = entry
        if self._is_writing_overloaded(0.01):
            # print(f"WARNING: delay writing {req.rid}")
            entry.is_synced = False
            self.writing_records[req.rid] = Queue()
        else:
            self._write_host(entry, req)
    
    def sync_unsynced_reqs(self):
        """Tries to synchronize unsynchronized requests."""
        with self.entries_lock:
            unsynced_entries = [
                entry for entry in self.entries.values() 
                if (
                    not entry.is_synced and
                    entry.value is not None and
                    entry.rid not in self.req_to_remove and
                    entry.rid not in self.req_to_evict
                )
            ]
            # sort the requests by their sequence length
            unsynced_entries.sort(key=lambda x: x.value.shape[0])
            for entry in unsynced_entries:
                if self._is_writing_overloaded():
                    break
                # print(f"INFO: try to synchronize {entry.rid} len: {entry.value.shape[0]}")
                self._write_host(entry, entry.req)
                entry.is_synced = True

    def _is_writing_overloaded(self, threshold: float = 0.02) -> bool:
        return self.write_token_num - self.wrote_token_num > threshold * self.write_speed

    def sync_batch(self, batch: ScheduleBatch):
        seq_lens_cpu = batch.seq_lens.cpu()
        if self.kv_selector is not None:
            self.kv_selector.update_with_batch(batch, self.req_to_remove)
        with self.entries_lock:
            for i, req in enumerate(batch.reqs):
                if req.rid in self.req_to_remove:
                    record = self.writing_records.get(req.rid)
                    if record is None or self.writing_records[req.rid].empty():
                        del self.req_to_remove[req.rid]
                        if record is not None:
                            del self.writing_records[req.rid]
                    continue
                if batch.forward_mode.is_extend():
                    self._sync_prefill(req, seq_lens_cpu[i])
                else:
                    self._sync_decode(req, seq_lens_cpu[i])
        # try to sync the unsynced requests
        self.sync_unsynced_reqs()
    
    def _initial_profile(self) -> Tuple[float, float]:
        PROFILE_LEN = 1024
        device_indices = self.token_to_kv_pool.alloc(PROFILE_LEN)
        if device_indices is None:
            raise RuntimeError("Failed to allocate device memory for profiling")
        host_indices = self.token_to_kv_pool_host.alloc(PROFILE_LEN)
        if host_indices is None:
            raise RuntimeError("Failed to allocate host memory for profiling")
        data = self.token_to_kv_pool_host.get_flat_data(host_indices)
        data = data.contiguous().pin_memory()
        start_time = time.time()
        self.token_to_kv_pool.transfer(device_indices, data)
        time_load = time.time() - start_time
        data = self.token_to_kv_pool.get_flat_data(device_indices)
        start_time = time.time()
        self.token_to_kv_pool_host.transfer(host_indices, data)
        time_write = time.time() - start_time
        # free the resources
        self.token_to_kv_pool.free(device_indices)
        self.token_to_kv_pool_host.free(host_indices)
        # return the throughput
        return PROFILE_LEN / time_load, PROFILE_LEN / time_write
