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

class MyScheduleDecision():
    def __init__(self, token_to_kv_pool, cum_buffer, output_speed,
                 running_reqs, waiting_reqs, loading_reqs,
                 max_running_requests, max_prefill_tokens, new_token_ratio):
        self.token_to_kv_pool = token_to_kv_pool
        self.cum_buffer = cum_buffer
        self.output_speed = output_speed

        self.keep_running_list = []
        self.new_load_list = []
        self.new_prefill_list = []

        self.running_reqs = running_reqs
        self.waiting_reqs = waiting_reqs
        self.loading_reqs = loading_reqs

        loading_tokens = 0
        loading_reqs = 0
        for req in self.loading_reqs:
            loading_reqs += 1
            # 我们假定loading的新请求至少需要多decode一秒钟的token量
            loading_tokens += len(req.output_ids) + len(req.origin_input_ids) + self.output_speed[req.rid]

        self.max_tokens = token_to_kv_pool.size - loading_tokens
        self.max_running_requests = max_running_requests - loading_reqs
        self.max_prefill_tokens = max_prefill_tokens

        self.avail_tokens = self.max_tokens
        self.avail_running_requests = self.max_running_requests
        self.avail_prefill_tokens = self.max_prefill_tokens

        self.new_token_ratio = new_token_ratio

        # self.initialize_keep_running_list()
    
    def estimate_req_kv_budget(self, req):
        min_generated_num = 32
        return (
            len(req.origin_input_ids) + len(req.output_ids) + 
            max(self.output_speed[req.rid] - self.cum_buffer[req.rid], min_generated_num)
        )

    def initialize_keep_running_list(self):
        # 默认我们认为调度策略就是沿用之前的running_batch不做任何改变
        self.running_reqs = sorted(self.running_reqs, key=lambda x: self.cum_buffer[x.rid])
        for req in self.running_reqs:
            kv_budget = self.estimate_req_kv_budget(req)
            if self.avail_running_requests > 0 and self.avail_tokens >= kv_budget:
                self.keep_running_list.append(req)
                self.avail_running_requests -= 1
                self.avail_tokens -= kv_budget

    def remove_request(self, req):
        if req not in self.keep_running_list and req not in self.new_load_list and req not in self.new_prefill_list:
            assert False, "Try to remove request not exist"
        
        if req in self.keep_running_list:
            self.keep_running_list.remove(req)
        elif req in self.new_prefill_list:
            self.new_prefill_list.remove(req)
            self.avail_prefill_tokens -= len(req.origin_input_ids) + len(req.output_ids)
        else:
            self.new_load_list.remove(req)
        
        self.avail_running_requests += 1
        self.avail_tokens += self.estimate_req_kv_budget(req)
    
    def can_add_request(self, req, recompute=False):
        if req in self.keep_running_list or req in self.new_load_list or req in self.new_prefill_list:
            return False
        
        if req in self.loading_reqs:
            return False
        elif req in self.waiting_reqs:
            if self.avail_running_requests > 0 and self.avail_tokens >= self.estimate_req_kv_budget(req):
                if recompute and self.avail_prefill_tokens >= len(req.origin_input_ids) + len(req.output_ids):
                    return True
                elif not recompute:
                    return True
                return False
        else:
            if self.avail_running_requests > 0 and self.avail_tokens >= self.estimate_req_kv_budget(req):
                return True
            return False

    def add_request(self, req, recompute=False):
        # 为外部的其余调度逻辑添加的接口
        if req in self.keep_running_list or req in self.new_load_list or req in self.new_prefill_list:
            assert False, "Duplicate add request"
        kv_budget = self.estimate_req_kv_budget(req)
        if req in self.loading_reqs:
            assert False, "Cannot add a request already in loading state"
        elif req in self.waiting_reqs:
            if self.avail_running_requests > 0 and self.avail_tokens >= kv_budget:
                if recompute and self.avail_prefill_tokens >= len(req.origin_input_ids) + len(req.output_ids):
                    self.new_prefill_list.append(req)
                    self.avail_prefill_tokens -= len(req.origin_input_ids) + len(req.output_ids)
                elif not recompute:
                    self.new_load_list.append(req)
                self.avail_running_requests -= 1
                self.avail_tokens -= kv_budget
        else:
            if self.avail_running_requests > 0 and self.avail_tokens >= kv_budget:
                self.keep_running_list.append(req)
                self.avail_running_requests -= 1
                self.avail_tokens -= kv_budget

class MyRequestOffloadManager():
    def __init__(self, sync_cache):
        self.load_queue = []
        self.loading_queue = []

        self.evict_queue = []
        self.sync_cache = sync_cache
    
    def add_load_request(self, req):
        #assert req not in self.load_queue, "Double adding the loading request"
        self.load_queue.append((req, time.time()))
    
    def add_evict_request(self, req, evict_len):
        #assert req not in self.evict_queue, "Double adding the evicting request"
        self.evict_queue.append((req, evict_len, time.time()))

    def get_total_reqs(self):
        return len(self.load_queue) + len(self.loading_queue) + len(self.evict_queue)

    def get_and_remove_finished_load(self):
        loaded_req = []
        for req in self.loading_queue:
            if self.sync_cache.load_check(req[0]):
                loaded_req.append(req)
        for req in loaded_req:
            self.loading_queue.remove(req)
        res = []
        for req in loaded_req:
            res.append(req[0])
        return res

    def is_all_evict_finished(self):
        return len(self.sync_cache.get_evicting_reqs()) == 0
    
    def is_before_evict_finished(self, time):
        for req in self.evict_queue:
            if req[2] < time:
                return False
        return True
    
    def get_ongoing_evict(self):
        return self.evict_queue
    
    def get_ongoing_load(self):
        return self.loading_queue
    
    def get_waiting_load(self):
        return self.load_queue

    def get_all_load_erqs(self):
        res = []
        for req in self.load_queue:
            res.append(req[0])
        for req in self.loading_queue:
            res.append(req[0])
        return res

    def step(self):
        if len(self.load_queue) > 0 or len(self.evict_queue) > 0:
            print(f"load queue: {len(self.load_queue)} evict queue: {len(self.evict_queue)}")
        # NOTE: Later we can adopt more fine-grained offload strategy
        # First we deal with the evict request, only request are finished we can only do loading
        finished_evict = []
        for req in self.evict_queue:
            self.sync_cache.evict_device(req[0], req[1])

        filtered_in_evict_queue = []
        finished_evict = []
        running_evict_reqs = self.sync_cache.get_evicting_reqs()
        for req in self.evict_queue:
            if req[0] not in running_evict_reqs:
                filtered_in_evict_queue.append(req)
        for req in filtered_in_evict_queue:
            self.evict_queue.remove(req)
            finished_evict.append(req[0])
        
        if len(self.load_queue) > 0:
            filtered_in_load_queue = []
            for req in self.load_queue:
                if self.is_before_evict_finished(req[1]):
                    # assert self.sync_cache.can_load_back(req[0]), "A request cannot loadback in the Offload Manager"
                    self.sync_cache.load_back(req[0])
                    filtered_in_load_queue.append(req)
            for req in filtered_in_load_queue:
                self.load_queue.remove(req)
                self.loading_queue.append(req)
        return finished_evict

class MyScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We force the scheduler to use CPU-GPU Radix Cache
        # self.sync_cache = ChunkCache(
        #     req_to_token_pool=self.req_to_token_pool,
        #     token_to_kv_pool=self.token_to_kv_pool,
        # )
        self.tree_cache = SyncChunkCache(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
        )
        self.kv_selector = self.tree_cache.init_kv_selector()

        # Some record
        self.decode_time_stamp = {}
        self.loading_time_stamp = {}
        self.rebuffer_time = {}
        self.output_speed = {}

        # Some information scheduler need
        self.next_prefill_batch = None
        self.cum_buffer_size = {}
        self.last_schedule = None
        self.reschedule = False
        self.schedule_decision = None
        self.last_running_time = None
        self.offload_manager = MyRequestOffloadManager(self.tree_cache)

        # Some custom scheduler config
        self.high_watermark_ratio = 5.0
        self.low_watermark_ratio = 1.0
        self.reschedule_interval = 1.0
        self.log_batch_status = True
        self.runtime_check = False
        self.debug_log = True
    
    @torch.no_grad()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        time_stamp = 0.0
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            
            torch.cuda.synchronize()
            st = time.time()

            batch = self.get_next_batch_to_run()
            if self.kv_selector is not None:
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

                    result = self.run_batch(batch)
                    torch.cuda.synchronize()
                    ed = time.time()
                    time_stamp += ed - st
                    self.last_running_time = ed - st
                    print(f"{ed - st} {time_stamp}", file=open('tmp/batch_detail.txt', 'a'))
                    print(f"{ed - st}", file=open('tmp/batch_info.txt', 'a'))
                else:
                    result = self.run_batch(batch)
                    torch.cuda.synchronize()
                    ed = time.time()
                    # if batch.forward_mode.is_decode():
                    #     if self.avg_decode_time == 0.0:
                    #         self.avg_decode_time = (ed - st)
                    #     else:
                    #         self.avg_decode_time = (self.avg_decode_time + (ed - st)) / 2
                    # elif batch.forward_mode.is_extend():
                    #     if self.avg_prefill_time == 0.0:
                    #         self.avg_prefill_time = (ed - st)
                    #     else:
                    #         self.avg_prefill_time = (self.avg_prefill_time + (ed - st)) / 2

                self.process_batch_result(batch, result)
            else:
                # Self-check and re-init some states when the server is idle
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
    
    def handle_generate_request(self, recv_req):
        self.cum_buffer_size[recv_req.rid] = 0
        self.rebuffer_time[recv_req.rid] = 0.0
        self.output_speed[recv_req.rid] = random.choice([5.0, 10.0])
        print(f'{recv_req.rid} {self.output_speed[recv_req.rid]}', file=open('tmp/output_speed_info.log', 'a'))
        # self.output_speed[recv_req.rid] = 10.0
        return super().handle_generate_request(recv_req)

    def update_buffer_size(self):
        # update cum_buffer_size and rebuffer time
        during_time = self.last_running_time
        if during_time is not None:
            for rid in self.cum_buffer_size:
                if self.cum_buffer_size[rid] - during_time * self.output_speed[rid] >= 0:
                    self.cum_buffer_size[rid] -= during_time * self.output_speed[rid]
                else:
                    self.rebuffer_time[rid] += during_time - \
                        self.cum_buffer_size[rid] / self.output_speed[rid]
                    self.cum_buffer_size[rid] = 0
            # print(self.cum_buffer_size)

    def get_valid_throughput(self):
        # currently we use e^{-x} as the valid throughput
        # e.g. if cum_buffer_size = 0, then throughput = 1
        # the larger the cum_buffer_size, the smaller the throughput
        valid_thr = {}
        if self.waiting_queue is not None:
            for req in self.waiting_queue:
                assert req.rid in self.cum_buffer_size, \
                    "Request not in cum_buffer_size, but it should be added when the request is received."
                valid_thr[req] = math.exp(-self.cum_buffer_size[req.rid])
        
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                assert req.rid in self.cum_buffer_size, \
                    "Request not in cum_buffer_size, but it should be added when the request is received."
                valid_thr[req] = math.exp(-self.cum_buffer_size[req.rid])
        
        return valid_thr

    def estimate_load_cost(self, req):
        return (len(req.origin_input_ids) + len(req.output_ids)) / 2048 

    def estimate_evict_cost(self, req):
        return self.estimate_load_cost(req) / 2
    
    def estimate_recompute_cost(self,req):
        if len(req.output_ids) == 0:
            return 0
        else:
            return self.estimate_load_cost(req) * 2

    def evaluate_objective(self, schedule_decision, value):
        total_value = 0
        evicted_reqs = set(schedule_decision.running_reqs) - set(schedule_decision.keep_running_list)
        evicted_tokens = [(len(req.origin_input_ids) + len(req.output_ids)) for req in evicted_reqs]
        load_tokens = [(len(req.origin_input_ids) + len(req.output_ids)) for req in schedule_decision.new_load_list]
        prefill_tokens = [(len(req.origin_input_ids) + len(req.output_ids)) for req in schedule_decision.new_prefill_list]

        for req in schedule_decision.keep_running_list:
            total_value += value.get(req, 0)
        for req in schedule_decision.new_load_list:
            total_value += value.get(req, 0)
        for req in schedule_decision.new_prefill_list:
            total_value += value.get(req, 0)

        return total_value - sum(evicted_tokens) / 4096 - max(sum(load_tokens) / 2048, sum(prefill_tokens) / 1024)

    def initial_greedy_selection(self, schedule_decision, value):
        candidates = []

        for req in schedule_decision.running_reqs:
            v = value.get(req, 0)
            adjusted_value = v  # running继续保留没有额外代价
            size = len(req.origin_input_ids) + len(req.output_ids) + max(self.output_speed[req.rid] - self.cum_buffer_size[req.rid], 0)
            candidates.append((req, adjusted_value, size, 'running', False))

        for req in schedule_decision.waiting_reqs:
            v = value.get(req, 0)
            size = len(req.origin_input_ids) + len(req.output_ids) + max(self.output_speed[req.rid] - self.cum_buffer_size[req.rid], 0)

            # 预先计算两种可能
            # prefilling方式
            recompute_cost = self.estimate_recompute_cost(req)
            adjusted_value_recompute = v # - recompute_cost
            candidates.append((req, adjusted_value_recompute, size, 'waiting', True))  # recompute=True
            
            if self.tree_cache.can_load_back(req):
                # 直接load方式
                load_cost = self.estimate_load_cost(req)
                adjusted_value_load = v - load_cost
                candidates.append((req, adjusted_value_load, size, 'waiting', False))  # recompute=False

        # 按 adjusted value / size 排序，性价比高的优先
        candidates.sort(key=lambda x: -x[1])

        for req, adjusted_value, size, source, recompute in candidates:
            if source == 'running':
                if schedule_decision.can_add_request(req):
                    schedule_decision.add_request(req)
            else:  # waiting
                if schedule_decision.can_add_request(req, recompute=recompute):
                    schedule_decision.add_request(req, recompute=recompute)
    
    def local_search(self, schedule_decision, value):
        improved = True

        while improved:
            improved = False
            current_objective = self.evaluate_objective(schedule_decision, value)

            all_reqs = list(schedule_decision.keep_running_list) + \
                        list(schedule_decision.new_load_list) + \
                        list(schedule_decision.new_prefill_list)

            # 尝试删除一个，加一个
            for remove_req in all_reqs:
                for try_add_req in schedule_decision.running_reqs + schedule_decision.waiting_reqs:
                    if try_add_req in all_reqs or try_add_req in schedule_decision.loading_reqs:
                        continue  # 跳过已在的或loading中的

                    # 尝试移除remove_req
                    schedule_decision.remove_request(remove_req)

                    # 尝试添加try_add_req
                    can_recompute = (try_add_req in schedule_decision.waiting_reqs)
                    if schedule_decision.can_add_request(try_add_req, recompute=can_recompute):
                        schedule_decision.add_request(try_add_req, recompute=can_recompute)

                        new_objective = self.evaluate_objective(schedule_decision, value)
                        if new_objective > current_objective:
                            # 接受更优结果
                            current_objective = new_objective
                            improved = True
                            break
                        else:
                            # 恢复状态
                            schedule_decision.remove_request(try_add_req)
                            schedule_decision.add_request(remove_req, recompute=(remove_req in schedule_decision.waiting_reqs))
                    else:
                        # 恢复状态
                        schedule_decision.add_request(remove_req, recompute=(remove_req in schedule_decision.waiting_reqs))
                if improved:
                    break
    
    def generate_schedule_decision(self):
        valid_thr = self.get_valid_throughput()

        schedule_decision = MyScheduleDecision(self.token_to_kv_pool, self.cum_buffer_size, self.output_speed,
                                            [] if self.running_batch is None else self.running_batch.reqs, 
                                            self.waiting_queue, self.offload_manager.get_all_load_erqs(),
                                            self.max_running_requests, self.max_prefill_tokens, self.new_token_ratio)

        self.initial_greedy_selection(schedule_decision, valid_thr)
        # self.local_search(schedule_decision, valid_thr)
        print('keep_running_list', schedule_decision.keep_running_list, file=open('tmp/schedule_output.txt', "a"))
        print('new_load_list', schedule_decision.new_load_list, file=open('tmp/schedule_output.txt', "a"))
        print('new_prefill_list', schedule_decision.new_prefill_list, file=open('tmp/schedule_output.txt', "a"))
        print('--------------------------------', file=open('tmp/schedule_output.txt', "a"))
        return schedule_decision
            
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

        # update_sync_cache()
        if self.last_batch and isinstance(self.tree_cache, SyncChunkCache):
            self.tree_cache.sync_batch(self.last_batch)
    
        # update_batch()
        if self.running_batch is not None:
            self.running_batch.filter_batch()
        
        # update_offload_manager()
        self.waiting_queue.extend(self.offload_manager.step())
        loaded_req_list = self.offload_manager.get_and_remove_finished_load()
        if len(loaded_req_list) != 0:
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
            if self.running_batch is None: # the running batch is empty, weird
                self.running_batch = loaded_batch
            else:
                self.running_batch.merge_batch(loaded_batch)

        # update_buffer_size()
        self.update_buffer_size()
        
        if (self.last_schedule is None or time.time() - self.last_schedule >= self.reschedule_interval) and self.offload_manager.is_all_evict_finished():
            if self.next_prefill_batch is not None:
                print('delayed prefill now')
                ret = self.next_prefill_batch
                self.next_prefill_batch = None
                ret.prepare_for_extend()
                return ret
            if self.running_batch is not None:
                seq_lens_cpu = self.running_batch.seq_lens.cpu().numpy()
                before_req_nums = len(self.running_batch.reqs) + len(self.waiting_queue) + self.offload_manager.get_total_reqs()
            else:
                before_req_nums = len(self.waiting_queue) + self.offload_manager.get_total_reqs()
            self.last_schedule = time.time()
            
            self.schedule_decision = self.generate_schedule_decision()
            
            if len(self.schedule_decision.new_prefill_list) == 0 and len(self.schedule_decision.keep_running_list) == 0:
                return None

            # first we get the evict request list and filter the running batch
            evict_list = []
            if self.running_batch is not None:
                assert len(self.running_batch.reqs) == len(set(self.running_batch.reqs)), \
                    "running batch has duplicate requests"
                for i, req in enumerate(self.running_batch.reqs):
                    if req not in self.schedule_decision.keep_running_list:
                        evict_list.append(req)
                
                # second we deal with the evict process
                for i, req in enumerate(evict_list):
                    idx = self.running_batch.reqs.index(req)
                    if isinstance(self.tree_cache, SyncChunkCache):
                        print(f'evict {req.rid} for {seq_lens_cpu[idx]} tokens')
                        self.offload_manager.add_evict_request(req, seq_lens_cpu[idx])
                    elif isinstance(self.tree_cache, ChunkCache):
                        # ChunkCache directly evict all tokens
                        token_indices = self.req_to_token_pool.req_to_token[
                            req.req_pool_idx, : seq_lens_cpu[idx]
                        ]
                        self.token_to_kv_pool.free(token_indices)
                        self.req_to_token_pool.free(req.req_pool_idx)
                        if req.rid in self.tree_cache.entries:
                            del self.tree_cache.entries[req.rid]
                        req.reset_for_retract()
                    else:
                        assert False, "Only ChunkCache and SyncChunkCache supports new scheduler"
                
                keep_indices = []
                for req in self.schedule_decision.keep_running_list:
                    idx = self.running_batch.reqs.index(req)
                    keep_indices.append(idx)
                self.running_batch.filter_batch(keep_indices=keep_indices)
                self.waiting_queue.extend(self.offload_manager.step())

                if self.running_batch.is_empty():
                    self.running_batch = None
            
            if len(self.schedule_decision.new_load_list) != 0:
                # first we filter the waiting queue
                self.waiting_queue = [
                    x for x in self.waiting_queue if x not in set(self.schedule_decision.new_load_list)
                ]
                # second we deal with the load process
                for i, req in enumerate(self.schedule_decision.new_load_list):
                    print(f"loading {req.rid}")
                    self.offload_manager.add_load_request(req)
                self.waiting_queue.extend(self.offload_manager.step())
            
            if len(self.schedule_decision.new_prefill_list) != 0:
                # first we filter the waiting queue
                self.waiting_queue = [
                    x for x in self.waiting_queue if x not in set(self.schedule_decision.new_prefill_list)
                ]
                
                for req in self.schedule_decision.new_prefill_list:
                    if req.rid in self.tree_cache.entries:
                        req.reset_for_retract()
                        # self.tree_cache.remove_req(req)
                    req.init_next_round_input(None)
                
                # else, we have to do recompute
                self.next_prefill_batch = ScheduleBatch.init_new(
                    self.schedule_decision.new_prefill_list,
                    self.req_to_token_pool,
                    self.token_to_kv_pool,
                    self.tree_cache,
                    self.model_config,
                    self.enable_overlap,
                    self.spec_algorithm,
                    self.server_args.enable_custom_logit_processor,
                    self.server_args.return_hidden_states,
                )

                if self.offload_manager.is_all_evict_finished():
                    print('directly do prefill after evict')
                    ret = self.next_prefill_batch
                    self.next_prefill_batch = None
                    ret.prepare_for_extend()
                else:
                    if self.running_batch is None:
                        ret = None
                    else:
                        self.running_batch = self.update_running_batch(self.running_batch)
                        ret = self.running_batch

                # check while scheduling, no request is lost
                current_req_nums = len(self.schedule_decision.new_prefill_list) + \
                    ((len(self.running_batch.reqs)) if (self.running_batch is not None) else 0) + \
                    ((len(self.waiting_queue)) if (self.waiting_queue is not None) else 0) + \
                    self.offload_manager.get_total_reqs()
                try:
                    assert current_req_nums == before_req_nums, \
                        f"request number not match ({current_req_nums} {before_req_nums})"
                except Exception as e:
                    print(len(self.schedule_decision.new_prefill_list), len(self.running_batch.reqs), len(self.waiting_queue))
                    raise e
            else:
                # check while scheduling, no request is lost
                current_req_nums = \
                    ((len(self.running_batch.reqs)) if (self.running_batch is not None) else 0) + \
                    ((len(self.waiting_queue)) if (self.waiting_queue is not None) else 0) + \
                    self.offload_manager.get_total_reqs()
                try:
                    assert current_req_nums == before_req_nums, \
                        f"request number not match ({current_req_nums} {before_req_nums})"
                except Exception as e:
                    print(len(self.schedule_decision.new_prefill_list), self.running_batch, self.waiting_queue)
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
