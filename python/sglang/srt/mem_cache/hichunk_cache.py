import heapq
import logging
import time
from typing import List, Optional

import torch

from python.sglang.srt.managers.schedule_batch import Req
from python.sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.memory_pool import (
    BaseTokenToKVPool,
    MLATokenToKVPoolHost,
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
            enable_memory_saver=req_to_token_pool.enable_memory_saver,
        )
        self.cache_controller = HiCacheController(
            token_to_kv_pool, self.token_to_kv_pool_host
        )
        super().__init__(req_to_token_pool, self.token_to_kv_pool_host)
    
    def reset(self):
        super().reset()
        self.token_to_kv_pool_host.reset()
        self.req_to_token_pool_host.reset()

    def write_backup(self, req: Req):
        req_pool_idx = self.req_to_token_pool_host.alloc(1)
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
            self.req_to_token_pool_host[req_pool_idx[0], :len(host_indices)] = host_indices
            entry.backuped = True
            entry.host_value = host_indices
            entry.host_req_pool_idx = req_pool_idx[0]
            req.last_node = entry
    
    def evict_host(self, num_tokens: int):
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
                backup_indices = self.req_to_token_pool_host.req_to_token[
                    entry.host_req_pool_idx, :len(entry.host_value)
                ]
                self.token_to_kv_pool_host.free(backup_indices)
                self.req_to_token_pool_host.free(entry.host_req_pool_idx)

                num_evicted += len(node.host_value)
                entry.host_value = None
                entry.backuped = False
                entry.host_req_pool_idx = None
                entry.host_value = None
        
        if num_evicted < num_tokens:
            raise RuntimeError(f"Failed to evict {num_tokens} tokens")

    def load_back(
        self, rid
    ):
        entry = self.entries[rid]
        if entry is None:
            raise RuntimeError(f"Failed to find rid {rid} in entry")
        if not entry.backuped:
            raise RuntimeError(f"Entry {rid} is not backuped")

        host_indices = entry.host_value
        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=entry
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=entry
            )
        if device_indices is not None:
            # invalidate the backup version
            backup_indices = self.req_to_token_pool_host.req_to_token[
                    entry.host_req_pool_idx, :len(entry.host_value)
                ]
            self.token_to_kv_pool_host.free(backup_indices)
            self.req_to_token_pool_host.free(entry.host_req_pool_idx)

            entry.value = device_indices
            entry.evicted = False
            entry.backup = False

            return device_indices
        
        return None

    def init_load_back(
        self,
        last_node: int,
        prefix_indices: List[int],
        mem_quota: Optional[int] = None,
    ):
        assert (
            len(prefix_indices) == 0
        ), "load back should be called with empty prefix"
         
        if last_node.evicted:
            loading_values = self.load_back(last_node.rid)
            if loading_values is not None:
                prefix_indices = loading_values
                logger.debug(
                    f"loading back {len(loading_values)} tokens for request {last_node.id}"
                )
        
        return self.entries[last_node.key], prefix_indices

    def evict_req(self, req: Req):
        # write back to the host memory
        if req.last_node is not None:
            self.write_backup(req)
        
        # invalidate the device memory
        device_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :len(req.prefix_indices)
        ]
        self.token_to_kv_pool.free(device_kv_indices)
        self.req_to_token_pool.free(req.req_pool_idx)

        # maintain the cache entry
        entry = self.entries[req.rid]
        entry.evicited = True
        entry.value = None
        