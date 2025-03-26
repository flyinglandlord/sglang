import heapq
import logging
import time
from typing import List, Optional

import torch

from python.sglang.srt.managers.schedule_batch import Req
from python.sglang.srt.mem_cache.chunk_cache import ChunkCache, ChunkCacheEntry
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.memory_pool import (
    BaseTokenToKVPool,
    MLATokenToKVPoolHost,
    MemoryStateInt,
    ReqToTokenPool,
)

logger = logging.getLogger(__name__)

class HiChunkCache(ChunkCache):

    def __init__(
        self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: BaseTokenToKVPool
    ):
        self.token_to_kv_pool_host = MLATokenToKVPoolHost(token_to_kv_pool)
        self.req_to_token_pool_host = ReqToTokenPool(
            req_to_token_pool.size,
            req_to_token_pool.max_context_len,
            device="cpu",
            enable_memory_saver=False,
        )
        self.cache_controller = HiCacheController(
            token_to_kv_pool, self.token_to_kv_pool_host,
            write_policy='write_back'
        )
        self.rid_to_req = {}
        # self.ongoing_load_back = {}
        self.ongoing_write = {}
        super().__init__(req_to_token_pool, token_to_kv_pool)
    
    def reset(self):
        super().reset()
        self.token_to_kv_pool_host.clear()
        self.req_to_token_pool_host.clear()
    
    def evict_host(self, num_tokens: int):
        print(f'need to evict {num_tokens} tokens from host memory')
        backuped_reqs = []
        for rid, node in self.entries.items():
            if node.backuped:
                backuped_reqs.append(rid)
        num_evicted = 0
        while num_evicted < num_tokens:
            if not backuped_reqs:
                break
            rid = backuped_reqs.pop(0)
            entry = self.entries[rid]
            if entry.backuped:
                if entry.host_value is None:
                    raise RuntimeError(f"Host value is None for rid {rid}")
                self.cache_controller.evict_host(entry.host_value)
                self.req_to_token_pool_host.free(entry.host_req_pool_idx)

                num_evicted += len(node.host_value)
                print(f'{rid} trigger evict from host memory')
                entry.host_value = None
                entry.backuped = False
                entry.host_req_pool_idx = None
                entry.host_value = None
                entry.req.reset_for_retract()
                del self.entries[rid]
        
        if num_evicted < num_tokens:
            raise RuntimeError(f"Failed to evict {num_tokens} tokens")
    
    def del_backup(self, rid):
        entry = self.entries[rid]
        # invalidate the backup version
        backup_indices = self.req_to_token_pool_host.req_to_token[
                entry.host_req_pool_idx, :len(entry.host_value)
            ]
        self.cache_controller.mem_pool_host.update_backup(backup_indices)
        self.cache_controller.evict_host(backup_indices)
        self.req_to_token_pool_host.free(entry.host_req_pool_idx)
        print('delete backup success')

    def load_back(
        self, rid
    ):
        entry = self.entries[rid]
        if entry is None:
            raise RuntimeError(f"Failed to find rid {rid} in entry")
        if not entry.backuped:
            raise RuntimeError(f"Entry {rid} is not backuped")

        host_indices = entry.host_value
        assert self.token_to_kv_pool_host.get_state(host_indices) == MemoryStateInt.BACKUP, \
            f"Host value is wrong for rid {rid}"
        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=entry
        )
        print(f'{rid} load back to {device_indices}')
        if device_indices is not None:
            entry.value = device_indices
            entry.evicted = False
            entry.backup = False
            return device_indices
        else:
            self.entries[rid].loading = False
            assert False, f"not enough device memory for {rid}, should check error in scheduler."
        return None

    def init_load_back(
        self,
        req,
        mem_quota: Optional[int] = None,
    ):
        # assert (
        #     len(prefix_indices) == 0
        # ), f"load back should be called with empty prefix, but got {len(prefix_indices)}"
         
        if req.last_node.evicted:
            print(f'{req.last_node.rid} triggers loading back')
            req.last_node.loading = True
            req.req_pool_idx = self.req_to_token_pool.alloc(1)[0]
            loading_values = self.load_back(req.rid)
            if loading_values is not None:
                print(
                    f"loading back {len(loading_values)} tokens for request {req.last_node.rid}"
                )
        
        # self.ongoing_load_back[req.last_node.rid] = req.last_node
        return self.entries[req.last_node.rid]

    def loading_check(self):
        while not self.cache_controller.ack_load_queue.empty():
            try:
                ack = self.cache_controller.ack_load_queue.get_nowait()
                # clear the reference
                self.entries[ack.rid].loading = False
                # del self.ongoing_load_back[ack.rid]
            except Exception:
                break

    def loading_complete(self, node):
        self.loading_check()
        return node.loading == False

    def cache_evicted_req(self, req: Req, seq_len: int):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :seq_len
        ]

        if req.rid not in self.entries:
            self.entries[req.rid] = ChunkCacheEntry(req.rid, kv_indices)

        entry = self.entries[req.rid]
        entry.value = kv_indices
        req.prefix_indices = []
        req.last_node = entry

    def write_backup(self, req: Req):
        req_pool_idx = self.req_to_token_pool_host.alloc(1)[0]
        if req_pool_idx is None:
            raise RuntimeError("Failed to allocate host memory for backup")
        
        entry = self.entries[req.rid]
        if entry is None:
            raise RuntimeError(f"Failed to find rid {req.rid} in entry")
        
        host_indices = self.cache_controller.write(
            device_indices=entry.value,
            node_id=entry
        )
        if host_indices is None:
            self.evict_host(len(entry.value))
            host_indices = self.cache_controller.write(
                device_indices=entry.value,
                node_id=entry
            )
        if host_indices is not None:
            print(f'{req.rid} write back success')
            self.req_to_token_pool_host.req_to_token[req_pool_idx, :len(host_indices)] = host_indices
            entry.backuped = True
            entry.host_value = host_indices
            entry.host_req_pool_idx = req_pool_idx
            req.last_node = entry
            self.ongoing_write[req.rid] = entry

    def evict_req(self, req: Req, evict_num: int):
        print(f'{req.rid} triggers eviction')
        entry = self.entries[req.rid]

        # write back to the host memory
        if req.last_node is not None:
            self.write_backup(req)
        
        # maintain the cache entry
        entry.req = req
        entry.evicted = True
        entry.value = None
        entry.evict_num = evict_num
        req.last_node = entry
    
    def writing_check(self):
        while not self.cache_controller.ack_write_queue.empty():
            try:
                ack = self.cache_controller.ack_write_queue.get_nowait()
                device_kv_indices = self.req_to_token_pool.req_to_token[
                    ack.req.req_pool_idx, :ack.evict_num
                ]
                self.cache_controller.evict_device(device_indices=device_kv_indices, host_indices=ack.host_value)
                self.req_to_token_pool.free(ack.req.req_pool_idx)
                # clear the reference
                del self.ongoing_write[ack.rid]
            except Exception:
                break
        