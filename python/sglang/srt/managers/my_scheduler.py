import faulthandler
import logging
import math
import os
import random
import signal
import threading
import time
import warnings
from collections import deque
from concurrent import futures
from types import SimpleNamespace
from typing import List, Optional

import psutil
import setproctitle
import torch
import zmq

from python.sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS_ESTIMATION, AddReqResult, PrefillAdder
from python.sglang.srt.mem_cache.sync_chunk_cache import SyncChunkCache
from python.sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from python.sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, get_bool_env_var, set_gpu_proc_affinity, suppress_other_loggers
from sglang.utils import get_exception_traceback
from sglang.srt.managers.scheduler import Scheduler
from sglang.global_config import global_config

logger = logging.getLogger(__name__)

class MyScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loading_queue = []

        self.log_batch_status = True
        self.decode_time_stamp = {}
        self.cum_buffer_size = {}
        self.rebuffer_time = {}
        self.output_speed = {}
        self.last_schedule = None
        self.reschedule = False
        self.last_update_buffer = None
        self.high_watermark_ratio = 5.0
        self.low_watermark_ratio = 1.0
        self.reschedule_interval = 1.0

        self.runtime_check = False
        self.debug_log = True

        # We force the scheduler to use CPU-GPU Radix Cache
        # self.tree_cache = ChunkCache(
        #     req_to_token_pool=self.req_to_token_pool,
        #     token_to_kv_pool=self.token_to_kv_pool,
        # )
        self.tree_cache = SyncChunkCache(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
        )
        self.kv_selector = self.tree_cache.init_kv_selector()
    
    @torch.no_grad()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        time_stamp = 0.0
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            if self.kv_selector:
                self.kv_selector.query_collector.reset()

            self.cur_batch = batch

            if batch:
                if self.log_batch_status:
                    for req in batch.reqs:
                        if (batch.decoding_reqs is not None and req not in batch.decoding_reqs) or batch.decoding_reqs is None:
                            print(f'{req.rid}', end=' ', file=open('tmp/batch_detail.txt', 'a'))
                    print('', file=open('tmp/batch_detail.txt', 'a'))
                    # for req in batch.reqs:
                    #     if (batch.decoding_reqs is not None and req not in batch.decoding_reqs) or batch.decoding_reqs is None:
                    #         print(f'{self.cum_buffer_size[req.rid]}', end=' ', file=open('tmp/batch_detail.txt', 'a'))
                    # print('', file=open('tmp/batch_detail.txt', 'a'))
                    if batch.decoding_reqs is not None:
                        for req in batch.decoding_reqs:
                            print(f'{req.rid}', end=' ', file=open('tmp/batch_detail.txt', 'a'))
                    print('', file=open('tmp/batch_detail.txt', 'a'))
                    # 2025.03.10: add the waiting queue information
                    if self.waiting_queue is not None:
                        for req in self.waiting_queue:
                            print(f'{req.rid}', end=' ', file=open('tmp/batch_detail.txt', 'a'))
                    print('', file=open('tmp/batch_detail.txt', 'a'))
                    if batch.forward_mode.is_mixed():
                        print(f"mixed: {len(batch.decoding_reqs)} {len(batch.reqs) - len(batch.decoding_reqs)} {sum(batch.prefix_lens)} {batch.extend_num_tokens}", end=' ', file=open('tmp/batch_info.txt', 'a'))
                    elif batch.forward_mode.is_extend():
                        print(f"extend: {0} {len(batch.reqs)} {sum(batch.prefix_lens)} {batch.extend_num_tokens}", end=' ', file=open('tmp/batch_info.txt', 'a'))
                    elif batch.forward_mode.is_decode():
                        print(f"decode: {len(batch.reqs)} {0} {batch.seq_lens_sum} {len(batch.reqs)}", end=' ', file=open('tmp/batch_info.txt', 'a'))
                    
                    torch.cuda.synchronize()
                    st = time.time()

                    result = self.run_batch(batch)

                    torch.cuda.synchronize()
                    ed = time.time()
                    time_stamp += ed - st
                    print(f"{ed - st} {time_stamp}", file=open('tmp/batch_detail.txt', 'a'))
                    print(f"{ed - st}", file=open('tmp/batch_info.txt', 'a'))
                else:
                    torch.cuda.synchronize()
                    st = time.time()

                    result = self.run_batch(batch)

                    torch.cuda.synchronize()
                    ed = time.time()

                    if batch.forward_mode.is_decode():
                        if self.avg_decode_time == 0.0:
                            self.avg_decode_time = (ed - st)
                        else:
                            self.avg_decode_time = (self.avg_decode_time + (ed - st)) / 2
                    elif batch.forward_mode.is_extend():
                        if self.avg_prefill_time == 0.0:
                            self.avg_prefill_time = (ed - st)
                        else:
                            self.avg_prefill_time = (self.avg_prefill_time + (ed - st)) / 2

                self.process_batch_result(batch, result)
            else:
                # Self-check and re-init some states when the server is idle
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
    
    def handle_generate_request(self, recv_req):
        self.cum_buffer_size[recv_req.rid] = 0
        self.rebuffer_time[recv_req.rid] = 0.0
        self.output_speed[recv_req.rid] = random.choice([40.0, 10.0])
        # self.output_speed[recv_req.rid] = 10.0
        return super().handle_generate_request(recv_req)

    def update_buffer_size(self):
        # update cum_buffer_size and rebuffer time
        if self.last_update_buffer is None:
            self.last_update_buffer = time.time()
        else:
            during_time = time.time() - self.last_update_buffer
            for rid in self.cum_buffer_size:
                # print(f"{rid} {self.cum_buffer_size[rid]} {during_time} {self.output_speed}")
                if self.cum_buffer_size[rid] - during_time * self.output_speed[rid] >= 0:
                    self.cum_buffer_size[rid] -= during_time * self.output_speed[rid]
                else:
                    self.rebuffer_time[rid] += during_time - \
                        self.cum_buffer_size[rid] / self.output_speed[rid]
                    self.cum_buffer_size[rid] = 0
            self.last_update_buffer = time.time()

    def get_valid_throughput(self):
        # currently we use e^{-x} as the valid throughput
        # e.g. if cum_buffer_size = 0, then throughput = 1
        # the larger the cum_buffer_size, the smaller the throughput
        res = {}
        if self.waiting_queue is not None:
            for req in self.waiting_queue:
                assert req.rid in self.cum_buffer_size, \
                    "Request not in cum_buffer_size, but it should be added when the request is received."
                res[req] = math.exp(-self.cum_buffer_size[req.rid])
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                assert req.rid in self.cum_buffer_size, \
                    "Request not in cum_buffer_size, but it should be added when the request is received."
                res[req] = math.exp(-self.cum_buffer_size[req.rid])
        return res
            
    def get_next_batch_to_run(self):
        # Merge the prefill batch into the running batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            self.reschedule = False
            if self.being_chunked_req:
                # Move the chunked request out of the batch
                self.last_batch.filter_batch(being_chunked_req=self.being_chunked_req)
                self.tree_cache.cache_unfinished_req(self.being_chunked_req)
                # being chunked request keeps its rid but will get a new req_pool_idx
                self.req_to_token_pool.free(self.being_chunked_req.req_pool_idx)
                self.batch_is_full = False

            if not self.last_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)
            if self.debug_log:
                print('trigger prefill batch merging, after merge prefill: ', file=open('tmp/debug_log.txt', 'a'))
                for req in self.running_batch.reqs:
                    print(f'{req}', file=open('tmp/debug_log.txt', 'a'))
                print(self.running_batch.seq_lens, file=open('tmp/debug_log.txt', 'a'))

        # write all updates to the sync cache
        if self.last_batch:
            self.tree_cache.sync_batch(self.last_batch)

        if self.running_batch is not None:
            self.running_batch.filter_batch()
        
        # check the loading queue
        if len(self.loading_queue) != 0:
            loaded_req_list = []
            for req in self.loading_queue:
                if self.tree_cache.load_check(req):
                    loaded_req_list.append(req)
            # print(self.sync_cache.req_write_op_count.values())
            self.loading_queue = [x for x in self.loading_queue if x not in set(loaded_req_list)]
            if self.debug_log:
                for req in loaded_req_list:
                    print(f'loaded request {req.rid}, length {len(req.output_ids) + len(req.origin_input_ids)}', file=open('tmp/mem_log.log', 'a+'))
            if len(loaded_req_list) != 0:
                # print('here we have loaded request now!')
                loaded_batch = ScheduleBatch.init_new(
                    loaded_req_list,
                    self.req_to_token_pool,
                    self.token_to_kv_pool,
                    self.tree_cache,
                    self.model_config,
                    self.enable_overlap,
                    self.spec_algorithm,
                    self.server_args.enable_custom_logit_processor,
                    self.server_args.return_hidden_states,
                )
                loaded_batch.prepare_for_resume_decode()
                if self.debug_log: print(f'merge loaded batch {loaded_batch}')
                if self.running_batch is None: # the running batch is empty, weird
                    self.running_batch = loaded_batch
                else:
                    self.running_batch.merge_batch(loaded_batch)
            if self.debug_log:
                print('trigger loaded batch merging, after merge loaded request: ', file=open('tmp/debug_log.txt', 'a'))
                for req in self.running_batch.reqs:
                    print(f'{req}', file=open('tmp/debug_log.txt', 'a'))
                print(self.running_batch.seq_lens, file=open('tmp/debug_log.txt', 'a'))
        
        # Do some check
        if self.runtime_check and self.running_batch is not None:
            print(self.running_batch)
            for i,req in enumerate(self.running_batch.reqs):
                # check the running request req_to_token_pool is not available
                if req.req_pool_idx in self.req_to_token_pool.free_slots:
                    print(req.req_pool_idx)
                    assert False, f"why running request {req.rid}'s req_pool_idx {req.req_pool_idx} is available??"
                
                # check running request token slot is released by error
                if torch.isin(self.token_to_kv_pool.free_slots, 
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :self.running_batch.seq_lens[i]].clone()
                .to(self.token_to_kv_pool.free_slots.device),).any():
                    print(self.req_to_token_pool.req_to_token[req.req_pool_idx, :self.running_batch.seq_lens[i]])
                    assert False, \
                    f"why running request {req.rid}'s token slots is in the free slots list??"
            
            # check if two running request share the token slots
            for i in range(len(self.running_batch.reqs)):
                for j in range(i+1, len(self.running_batch.reqs)):
                    if torch.isin(self.req_to_token_pool.req_to_token[self.running_batch.reqs[i].req_pool_idx, :self.running_batch.seq_lens[i]], 
                    torch.tensor(self.req_to_token_pool.req_to_token[self.running_batch.reqs[j].req_pool_idx, :self.running_batch.seq_lens[j]], 
                                    device=(self.req_to_token_pool.req_to_token.device))).any():
                        print(self.req_to_token_pool.req_to_token[self.running_batch.reqs[i].req_pool_idx, :self.running_batch.seq_lens[i]])
                        print(self.req_to_token_pool.req_to_token[self.running_batch.reqs[j].req_pool_idx, :self.running_batch.seq_lens[j]])
                        print('i', len(self.req_to_token_pool.req_to_token[self.running_batch.reqs[i].req_pool_idx, :self.running_batch.seq_lens[i]]))
                        print('j', len(self.req_to_token_pool.req_to_token[self.running_batch.reqs[j].req_pool_idx, :self.running_batch.seq_lens[j]]))
                        assert False, f"two running request {self.running_batch.reqs[i].rid} and {self.running_batch.reqs[j].rid} share the same token slots"

        self.update_buffer_size()
        
        if self.running_batch and len(self.running_batch.reqs) > 0:
            print(f'running size: {len(self.running_batch.reqs) if self.running_batch is not None else 0}, waiting size: {len(self.waiting_queue) if self.waiting_queue is not None else 0}, loading size: {len(self.loading_queue)}')

        if self.last_schedule is None or time.time() - self.last_schedule >= self.reschedule_interval:
            self.last_schedule = time.time()
            vthrouput = self.get_valid_throughput()

            # sort the request, by vthroughput and rebuffer time from high to low
            sorted_req = sorted(vthrouput.keys(), key=lambda x: vthrouput[x], reverse=True)
            before_req_nums = len(sorted_req) + len(self.loading_queue)

            # estimate the loading tokens, because we cannot evicted them immediately
            loading_tokens = 0
            loading_reqs = 0
            for req in self.loading_queue:
                loading_reqs += 1
                loading_tokens += len(req.output_ids) + len(req.origin_input_ids) + \
                    max(self.high_watermark_ratio * self.output_speed[req.rid] - self.cum_buffer_size[req.rid], 0)

            new_prefill_list = []
            keep_decode_list = []
            max_tokens = self.token_to_kv_pool.size - loading_tokens
            max_running_requests = self.max_running_requests - loading_reqs
            max_prefill_tokens = self.max_prefill_tokens
            new_token_ratio = self.new_token_ratio
            if self.running_batch is not None:
                seq_lens_cpu = self.running_batch.seq_lens.cpu().numpy()
            
            min_reserved_output_len = 16
            for req in sorted_req:
                if self.running_batch is not None and req in self.running_batch.reqs:
                    idx = self.running_batch.reqs.index(req)
                    remain_tokens = max_tokens - seq_lens_cpu[idx] - min(
                                (req.sampling_params.max_new_tokens - len(req.output_ids))* new_token_ratio,
                                max(self.high_watermark_ratio * self.output_speed[req.rid] - self.cum_buffer_size[req.rid], min_reserved_output_len),
                            ) 
                    if remain_tokens >= 0 and max_running_requests >= 1:
                        # print("selected", req.rid, max_tokens-remain_tokens, seq_lens_cpu[idx])
                        max_tokens = remain_tokens
                        keep_decode_list.append(req)
                        max_running_requests -= 1
                    else:
                        continue
                elif self.waiting_queue is not None and req in self.waiting_queue:
                    req.init_next_round_input(None)
                    # DEBUG NOTE: for resumed requests, we need to get the total token number including the generated tokens
                    req_len = len(req.fill_ids) if req.fill_ids else req.extend_input_len 
                    remain_tokens = max_tokens - req_len - min(
                                (req.sampling_params.max_new_tokens - len(req.output_ids))* new_token_ratio,
                                max(self.high_watermark_ratio * self.output_speed[req.rid] - self.cum_buffer_size[req.rid], min_reserved_output_len),
                            ) 
                    if remain_tokens >= 0 and \
                    max_prefill_tokens - req.extend_input_len >= 0 and \
                    max_running_requests >= 1:
                        max_tokens = remain_tokens
                        max_prefill_tokens -= req.extend_input_len
                        new_prefill_list.append(req)
                        max_running_requests -= 1
                    else:
                        continue
                else:
                    assert False, "code should not arrive here"
            
            if len(new_prefill_list) == 0 and len(keep_decode_list) == 0:
                return None
            
            # first sweep out the running batch request not in keep_decode_list
            swap_out = []
            if self.running_batch is not None:
                # check the running batch has no duplicate requests
                assert len(self.running_batch.reqs) == len(set(self.running_batch.reqs)), \
                    "running batch has duplicate requests"
                # get the running batch size before schedule
                before_running_batch = len(self.running_batch.reqs)

                for i, req in enumerate(self.running_batch.reqs):
                    # req.init_next_round_input(None)
                    if req not in keep_decode_list:
                        if isinstance(self.tree_cache, SyncChunkCache):
                            if self.debug_log: print(f"evict {req.rid} from device, length {seq_lens_cpu[i]}", file=open('tmp/mem_log.log', 'a+'))
                            try:
                                self.tree_cache.wait_write(req)
                                self.tree_cache.evict_device(req, seq_lens_cpu[i])
                            except Exception as e:
                                # for req in self.running_batch.reqs:
                                #     print(f'{req}')
                                # print(seq_lens_cpu)
                                raise e
                        elif isinstance(self.tree_cache, ChunkCache):
                            # ChunkCache directly evict all tokens
                            token_indices = self.req_to_token_pool.req_to_token[
                                req.req_pool_idx, : seq_lens_cpu[i]
                            ]
                            self.token_to_kv_pool.free(token_indices)
                            self.req_to_token_pool.free(req.req_pool_idx)
                            if req.rid in self.tree_cache.entries:
                                del self.tree_cache.entries[req.rid]
                            req.reset_for_retract()
                        else:
                            assert False, "Only ChunkCache and SyncChunkCache supports new scheduler"
                        swap_out.append(req)
                keep_indices = []
                keep_decode_kv_size = 0
                # self.tree_cache.writing_check()
                for req in keep_decode_list:
                    idx = self.running_batch.reqs.index(req)
                    keep_indices.append(idx)
                    keep_decode_kv_size += seq_lens_cpu[idx]
                self.running_batch.filter_batch(keep_indices=keep_indices)

                # check the req in swap_out is not in the running batch
                for req in swap_out:
                    assert req not in self.running_batch.reqs, f"request {req.rid} in running batch"
                assert len(self.running_batch.reqs) == len(keep_indices), \
                    f"filter batch function not working {len(self.running_batch.reqs)} {len(keep_indices)}"
                
                self.waiting_queue.extend(swap_out)

                # check the running batch size is correct
                after_running_batch = len(self.running_batch.reqs)
                try:
                    assert before_running_batch == after_running_batch + len(swap_out), \
                        f"running batch size not match {before_running_batch} {after_running_batch} {len(swap_out)}"
                except Exception as e:
                    print(keep_decode_list, swap_out, keep_indices)
                    raise e

                if self.running_batch.is_empty():
                    self.running_batch = None
            
            if len(new_prefill_list) != 0:
                self.waiting_queue = [
                    x for x in self.waiting_queue if x not in set(new_prefill_list)
                ]

                # check the total token number is valid (not exceed the GPU available size)
                total_new_token = 0
                for i in new_prefill_list:
                    total_new_token += i.extend_input_len
                try:
                    assert total_new_token <= self.token_to_kv_pool.available_size(), \
                        f"new token {total_new_token} exceed the available size {self.token_to_kv_pool.available_size()}"
                except Exception as e:
                    raise e
                
                # if request has been backup in cpu memory, just load it back
                can_load_reqs_in_prefill_list = []
                for req in new_prefill_list:
                    if self.tree_cache.can_load_back(req):
                        self.tree_cache.load_back(req)
                        # print(f'{req.rid} can load back, kv pool size: {self.token_to_kv_pool.available_size()}')
                        # new_prefill_list.remove(req)
                        can_load_reqs_in_prefill_list.append(req)
                        self.loading_queue.append(req)
                
                new_prefill_list = [x for x in new_prefill_list if x not in set(can_load_reqs_in_prefill_list)]
                for req in new_prefill_list:
                    # print(f'{req.rid} trigger recompute/prefill, kv pool size: {self.token_to_kv_pool.available_size()}, extend input len: {req.extend_input_len}')
                    if req.rid in self.tree_cache.entries:
                        req.reset_for_retract()
                        self.tree_cache.remove_req(req.rid)
                        print(f'{req.rid} remove from sync cache')
                    # req.init_next_round_input(None)
                # else, we have to do recompute
                ret = ScheduleBatch.init_new(
                    new_prefill_list,
                    self.req_to_token_pool,
                    self.token_to_kv_pool,
                    self.tree_cache,
                    self.model_config,
                    self.enable_overlap,
                    self.spec_algorithm,
                    self.server_args.enable_custom_logit_processor,
                    self.server_args.return_hidden_states,
                )
                ret.prepare_for_extend()

                # check ret.reqs request not share the same token slots
                if self.runtime_check:
                    for i in range(len(ret.reqs)):
                        for j in range(i+1, len(ret.reqs)):
                            if torch.isin(self.req_to_token_pool.req_to_token[ret.reqs[i].req_pool_idx, :ret.seq_lens[i]], 
                            self.req_to_token_pool.req_to_token[ret.reqs[j].req_pool_idx, :ret.seq_lens[j]].clone() 
                            .to(self.req_to_token_pool.req_to_token.device)).any():
                                print(self.tree_cache.entries.keys())
                                assert False, f"two prefill request {ret.reqs[i].rid} and {ret.reqs[j].rid} share the same token slots"

                    # check ret.output_loc is available
                    if torch.isin(self.token_to_kv_pool.free_slots, ret.out_cache_loc.clone()
                    .to(self.token_to_kv_pool.free_slots.device)).any():
                        print(len(ret.out_cache_loc))
                        assert False, \
                        f"output loc not available, {torch.isin(self.token_to_kv_pool.free_slots, torch.tensor(ret.out_cache_loc, device=self.token_to_kv_pool.free_slots.device))}"

                # check while scheduling, no request is lost
                current_req_nums = len(new_prefill_list) + \
                    ((len(self.running_batch.reqs)) if (self.running_batch is not None) else 0) + \
                    ((len(self.waiting_queue)) if (self.waiting_queue is not None) else 0) + \
                    ((len(self.loading_queue)) if (self.loading_queue is not None) else 0)
                try:
                    assert current_req_nums == before_req_nums, \
                        f"request number not match ({current_req_nums} {before_req_nums})"
                except Exception as e:
                    print(len(new_prefill_list), len(self.running_batch.reqs), len(self.waiting_queue))
                    raise e

                if ret.is_empty():
                    if self.running_batch is not None:
                        self.running_batch = self.update_running_batch(self.running_batch)
                    elif self.running_batch is None or self.running_batch.is_empty():
                        self.running_batch = None
                    ret = self.running_batch
            else:
                # check while scheduling, no request is lost
                current_req_nums = \
                    ((len(self.running_batch.reqs)) if (self.running_batch is not None) else 0) + \
                    ((len(self.waiting_queue)) if (self.waiting_queue is not None) else 0) + \
                    ((len(self.loading_queue)) if (self.loading_queue is not None) else 0)
                try:
                    assert current_req_nums == before_req_nums, \
                        f"request number not match ({current_req_nums} {before_req_nums})"
                except Exception as e:
                    print(len(new_prefill_list), self.running_batch, self.waiting_queue)
                    raise e
                
                self.running_batch = self.update_running_batch(self.running_batch)
                if self.running_batch is None:
                    ret = None
                else:
                    ret = self.running_batch
        else:
            if self.running_batch is None:
                ret = None
            else:
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch

        # Keep the old code
        # new_batch = self.get_new_batch_prefill()
        # if new_batch is not None:
        #     # Run prefill first if possible
        #     ret = new_batch
        # else:
        #     # Run decode
        #     if self.running_batch is None:
        #         ret = None
        #     else:
        #         self.running_batch = self.update_running_batch(self.running_batch)
        #        ret = self.running_batch

        # check if any request in both running batch and waiting queue
        if self.runtime_check:
            if self.running_batch is not None and self.waiting_queue is not None:
                for req in self.running_batch.reqs:
                    if req in self.waiting_queue:
                        assert False, "Request in both running batch and waiting queue"
        
        # Handle DP attention
        if self.server_args.enable_dp_attention:
            ret = self.prepare_dp_attn_batch(ret)
        
        if ret is not None and self.debug_log:
            print('----------------------', file=open('tmp/debug_log.txt', 'a'))
            for req in ret.reqs:
                print(f'{req}', file=open('tmp/debug_log.txt', 'a'))
            print(f'{self.tree_cache.entries.keys()}', file=open('tmp/debug_log.txt', 'a'))
            print(f'{ret.seq_lens}', file=open('tmp/debug_log.txt', 'a'))
            if self.running_batch is not None and ret != self.running_batch:
                print(f'Running batch: ', file=open('tmp/debug_log.txt', 'a'))
                for req in self.running_batch.reqs:
                    print(f'{req}', file=open('tmp/debug_log.txt', 'a'))
                print(f"{self.running_batch.seq_lens}", file=open('tmp/debug_log.txt', 'a'))
        return ret
    
    def get_new_batch_prefill(self):
        return super().get_new_batch_prefill()
    
    def process_batch_result_decode(self, batch: ScheduleBatch, result):
        # update the virtual buffer size & decode time stamp
        #for req in batch.reqs:
        #    self.virtual_buffer_size[req.rid] += 1
        for req in batch.reqs:
            if req.rid not in self.decode_time_stamp:
                self.decode_time_stamp[req.rid] = [time.time()]
            else:
                self.decode_time_stamp[req.rid].append(time.time())
            self.cum_buffer_size[req.rid] += 1
        super().process_batch_result_decode(batch, result)

    def process_batch_result_prefill(self, batch, result):
        for req in batch.reqs:
            if req.rid not in self.decode_time_stamp:
                self.decode_time_stamp[req.rid] = [time.time()]
            else:
                self.decode_time_stamp[req.rid].append(time.time())
            self.cum_buffer_size[req.rid] += 1
        return super().process_batch_result_prefill(batch, result)


def run_my_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    setproctitle.setproctitle("sglang::scheduler")
    faulthandler.enable()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configue the logger
    if dp_rank is None:
        configure_logger(server_args, prefix=f" TP{tp_rank}")
    else:
        configure_logger(server_args, prefix=f" DP{dp_rank} TP{tp_rank}")
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    parent_process = psutil.Process().parent()

    # Create a scheduler and run the event loop
    try:
        scheduler = MyScheduler(server_args, port_args, gpu_id, tp_rank, dp_rank)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )
        if scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
