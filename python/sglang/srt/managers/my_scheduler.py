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
                 running_reqs, waiting_reqs, loading_reqs, evicting_reqs, waiting_prefill_reqs,
                 max_running_requests, max_prefill_tokens, max_prefill_requests, new_token_ratio):
        self.token_to_kv_pool = token_to_kv_pool
        self.cum_buffer = cum_buffer
        self.output_speed = output_speed

        self.keep_running_list = []
        self.new_load_list = []
        self.new_prefill_list = []

        self.running_reqs = running_reqs
        self.waiting_reqs = waiting_reqs
        self.loading_reqs = loading_reqs
        self.evicting_reqs = evicting_reqs
        self.waiting_prefill_reqs = waiting_prefill_reqs
        # 我们假定loading的新请求最少需要decode出新的token量
        self.min_generated_num = 64

        loading_tokens = 0
        loading_reqs = 0
        for req in self.loading_reqs:
            loading_reqs += 1
            loading_tokens += len(req.output_ids) + len(req.origin_input_ids) + self.min_generated_num

        evicting_tokens = 0
        evicting_reqs = 0
        for req in self.evicting_reqs:
            evicting_reqs += 1
            evicting_tokens += len(req.output_ids) + len(req.origin_input_ids)

        prefill_tokens = 0
        prefill_reqs = 0
        for req in self.evicting_reqs:
            prefill_reqs += 1
            prefill_tokens += len(req.fill_ids)

        self.max_tokens = token_to_kv_pool.size - loading_tokens - evicting_tokens - prefill_tokens
        self.max_running_requests = max_running_requests - loading_reqs - evicting_reqs - prefill_reqs
        self.max_prefill_tokens = max_prefill_tokens
        self.max_prefill_requests = max_prefill_requests

        self.avail_tokens = self.max_tokens
        self.avail_running_requests = self.max_running_requests
        self.avail_prefill_tokens = self.max_prefill_tokens
        self.avail_prefill_requests = self.max_prefill_requests

        self.new_token_ratio = new_token_ratio

        # self.initialize_keep_running_list()
    
    def estimate_req_kv_budget(self, req):
        min_generated_num = 64
        return (
            len(req.origin_input_ids) + len(req.output_ids) + min_generated_num
        )

    def initialize_keep_running_list(self, available_tokens, available_requests):
        # 默认我们认为调度策略就是沿用之前的running_batch不做任何改变
        self.running_reqs = sorted(self.running_reqs, key=lambda x: self.cum_buffer[x.rid])
        self.avail_tokens -= len(self.running_reqs) * self.min_generated_num
        for req in self.running_reqs:
            if self.can_add_request(req):
                self.add_request(req)
        self.avail_running_requests = available_requests
        self.avail_tokens = available_tokens

    def remove_request(self, req):
        if req not in self.keep_running_list and req not in self.new_load_list and req not in self.new_prefill_list:
            assert False, "Try to remove request not exist"
        
        if req in self.keep_running_list:
            self.keep_running_list.remove(req)
        elif req in self.new_prefill_list:
            self.new_prefill_list.remove(req)
            self.avail_prefill_tokens += len(req.origin_input_ids) + len(req.output_ids)
            self.avail_prefill_requests += 1
        else:
            self.new_load_list.remove(req)
        
        self.avail_running_requests += 1
        self.avail_tokens += self.estimate_req_kv_budget(req)
    
    def can_add_request(self, req, recompute=False):
        # print(self.avail_prefill_tokens, self.avail_running_requests, self.avail_tokens, req)
        if req in self.keep_running_list or req in self.new_load_list or req in self.new_prefill_list:
            return False
        
        if req in self.loading_reqs:
            return False
        elif req in self.waiting_reqs:
            if self.avail_running_requests > 0 and self.avail_tokens >= self.estimate_req_kv_budget(req):
                if recompute and self.avail_prefill_tokens >= len(req.origin_input_ids) + len(req.output_ids) and self.avail_prefill_requests > 0:
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
                if recompute and self.avail_prefill_tokens >= len(req.origin_input_ids) + len(req.output_ids) and self.avail_prefill_requests > 0:
                    self.new_prefill_list.append(req)
                    self.avail_prefill_tokens -= len(req.origin_input_ids) + len(req.output_ids)
                    self.avail_prefill_requests -= 1
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
        self.evicting_queue = []
        self.sync_cache = sync_cache

        self.begin_time_stamp = {}
    
    def add_load_request(self, req):
        #assert req not in self.load_queue, "Double adding the loading request"
        self.load_queue.append((req, time.time()))
        self.begin_time_stamp[req.rid] = time.time()
    
    def add_evict_request(self, req, evict_len):
        #assert req not in self.evict_queue, "Double adding the evicting request"
        self.evict_queue.append((req, evict_len, time.time()))
        self.begin_time_stamp[req.rid] = time.time()

    def get_total_reqs(self):
        return len(self.load_queue) + len(self.loading_queue) + len(self.evict_queue) + len(self.evicting_queue)

    def get_and_remove_finished_load(self):
        loaded_req = []
        for req in self.loading_queue:
            if self.sync_cache.load_check(req[0]):
                loaded_req.append(req)
        for req in loaded_req:
            self.loading_queue.remove(req)
            print(f"{req[0].rid} with {len(req[0].origin_input_ids) + len(req[0].output_ids)} \
                  finished load in {time.time() - self.begin_time_stamp[req[0].rid]}", file=open('tmp/offload_log.log', 'a'))
            del self.begin_time_stamp[req[0].rid]
        res = []
        for req in loaded_req:
            res.append(req[0])
        return res

    def is_all_evict_finished(self):
        # (self.sync_cache.get_evicting_reqs())
        return len(self.sync_cache.get_evicting_reqs()) == 0
    
    def is_before_evict_finished(self, time):
        for req in self.evicting_queue:
            if req[2] < time:
                return False
        for req in self.evict_queue:
            if req[2] < time:
                return False
        return True
    
    def get_ongoing_evict(self):
        return self.evicting_queue
    
    def get_waiting_evict(self):
        return self.evict_queue
    
    def get_ongoing_load(self):
        return self.loading_queue
    
    def get_waiting_load(self):
        return self.load_queue

    def get_all_load_reqs(self):
        res = []
        for req in self.load_queue:
            res.append(req[0])
        for req in self.loading_queue:
            res.append(req[0])
        return res
    
    def get_all_evict_reqs(self):
        res = []
        for req in self.evict_queue:
            res.append(req[0])
        for req in self.evicting_queue:
            res.append(req[0])
        return res

    def is_req_ready_to_load(self, req):
        avail_reqs = self.sync_cache.req_to_token_pool.available_size()
        avail_tokens = self.sync_cache.token_to_kv_pool.available_size()

        if avail_reqs > 0 and avail_tokens > len(req.origin_input_ids) + len(req.output_ids) + 16:
            return True

        return False

    def step(self):
        if len(self.load_queue) > 0 or len(self.evict_queue) > 0:
            print(f"load wait queue: {len(self.load_queue)}, loading queue: {len(self.loading_queue)}, evicting queue: {len(self.evicting_queue)}", file=open('tmp/offload_log.log', 'a'))
            print('predict evict time', self.sync_cache.get_writing_workload(), file=open('tmp/offload_log.log', 'a'))
            print('predict load time', self.sync_cache.get_loading_workload(), file=open('tmp/offload_log.log', 'a'))
        # NOTE: Later we can adopt more fine-grained offload strategy
        # First we deal with the evict request, only request are finished we can only do loading
        #print(f"1: {len(self.evict_queue)}, {len(self.evicting_queue)}")

        for req in self.evict_queue:
            self.sync_cache.evict_device(req[0], req[1])
            self.evicting_queue.append(req)
        self.evict_queue = []
        #print(f"2: {len(self.evict_queue)}, {len(self.evicting_queue)}")

        # Second check the evicting queue exist any evicted reqs
        finished_evict = []
        new_evicting_queue = self.sync_cache.get_evicting_reqs()
        #print('evicting_queue len', len(new_evicting_queue))
        #print('evicting_queue', new_evicting_queue)
        finished_evict = [req[0] for req in self.evicting_queue if req[0].rid not in new_evicting_queue]
        for req in finished_evict:
            print(f"{req.rid} with {len(req.origin_input_ids) + len(req.output_ids)} \
                  finished evict in {time.time() - self.begin_time_stamp[req.rid]}", file=open('tmp/offload_log.log', 'a'))
            del self.begin_time_stamp[req.rid]
        self.evicting_queue = [req for req in self.evicting_queue if req[0].rid in new_evicting_queue]
        #print(f"3: {len(self.evict_queue)}, {len(finished_evict)} ,{len(self.evicting_queue)}")
        #print("available token slots", self.sync_cache.req_to_token_pool.available_size())
        
        # Finally perform loading if possible
        self.load_queue = sorted(self.load_queue, key=lambda x: (x[1]))
        if len(self.load_queue) > 0:
            filtered_in_load_queue = []
            for req in self.load_queue:
                if self.is_req_ready_to_load(req[0]):
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
        self.req_last_run_time = {}

        # Some custom scheduler config
        self.high_watermark_ratio = 5.0
        self.low_watermark_ratio = 1.0
        self.reschedule_interval = 1.0
        self.log_batch_status = False
        self.runtime_check = False
        self.debug_log = False
    
    @torch.no_grad()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        time_stamp = 0.0
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            
            #torch.cuda.synchronize()
            st = time.time()

            batch = self.get_next_batch_to_run()
            if self.kv_selector is not None:
                self.kv_selector.query_collector.reset()

            self.cur_batch = batch

            if batch:
                if self.log_batch_status:
                    result = self.run_batch(batch)
                    #torch.cuda.synchronize()
                    ed = time.time()
                    time_stamp += ed - st
                    self.last_running_time = ed - st

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

                    print(f"{ed - st} {time_stamp}", file=open('tmp/batch_detail.txt', 'a'))
                    print(f"{ed - st}", file=open('tmp/batch_info.txt', 'a'))
                else:
                    result = self.run_batch(batch)
                    #torch.cuda.synchronize()
                    ed = time.time()
                    self.last_running_time = ed - st
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
                self.tree_cache.cache_controller.can_write()
                self.process_batch_result(batch, result)
                self.tree_cache.cache_controller.dont_write()
            else:
                # Self-check and re-init some states when the server is idle
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
    
    def handle_generate_request(self, recv_req):
        self.cum_buffer_size[recv_req.rid] = 0
        self.rebuffer_time[recv_req.rid] = 0.0
        self.output_speed[recv_req.rid] = recv_req.sampling_params.output_speed
        # print(f'{recv_req.rid} {self.output_speed[recv_req.rid]}', file=open('tmp/output_speed_info.log', 'a'))
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

    def estimate_load_cost(self, req):
        loading_tokens, load_speed = self.tree_cache.get_loading_workload()
        return (loading_tokens + len(req.origin_input_ids) + len(req.output_ids)) / load_speed

    def estimate_evict_cost(self, req):
        return self.estimate_load_cost(req) / 2
    
    def estimate_recompute_cost(self, req):
        return (len(req.origin_input_ids) + len(req.output_ids)) * (0.12375879287719727 / 15388)

    def evaluate_objective(self, schedule_decision, value):
        # Calculate the final objective function
        total_value = 0
        current_evicting_tokens = self.tree_cache.get_writing_workload()[0]
        current_evicting_speed = self.tree_cache.get_writing_workload()[1]
        current_loading_tokens = self.tree_cache.get_loading_workload()[0]
        current_loading_speed = self.tree_cache.get_loading_workload()[1]

        all_reqs = schedule_decision.keep_running_list + schedule_decision.new_prefill_list + schedule_decision.new_load_list
        all_tokens = [(len(req.origin_input_ids) + len(req.output_ids)) for req in all_reqs]
        estimate_decode_time = sum(all_tokens) * (0.032536983489990234 / 20551)

        evicted_reqs = set(schedule_decision.running_reqs) - set(schedule_decision.keep_running_list)
        evicted_tokens = [(len(req.origin_input_ids) + len(req.output_ids)) for req in evicted_reqs]
        current_evicting_tokens = self.tree_cache.get_writing_workload()[0]
        current_evicting_speed = self.tree_cache.get_writing_workload()[1]
        estimate_evict_time = (current_evicting_tokens + sum(evicted_tokens)) / current_evicting_speed

        load_tokens = [(len(req.origin_input_ids) + len(req.output_ids)) for req in schedule_decision.new_load_list]
        estimate_load_time = (current_loading_tokens + sum(load_tokens)) / current_loading_speed

        prefill_tokens = [(len(req.origin_input_ids) + len(req.output_ids)) for req in schedule_decision.new_prefill_list]
        estimate_prefill_time = sum(prefill_tokens) * (0.12375879287719727 / 15388)

        real_reschedule_interval = max(self.reschedule_interval, 
                                       estimate_evict_time + estimate_load_time,
                                       estimate_evict_time + estimate_prefill_time)

        new_load_list = sorted(schedule_decision.new_load_list, key=lambda x: (len(x.origin_input_ids) + len(x.output_ids)))
        new_prefill_list = sorted(schedule_decision.new_prefill_list, key=lambda x: (len(x.origin_input_ids) + len(x.output_ids)))
        estimate_buffer_size = {}

        for req in schedule_decision.keep_running_list:
            current_buffer = self.cum_buffer_size[req.rid]

            total_value += value.get(req, 0) * real_reschedule_interval
            estimate_buffer_size[req.rid] = current_buffer \
                    + real_reschedule_interval / estimate_decode_time \
                    - real_reschedule_interval * self.output_speed[req.rid]
        
        cum_load_time = 0.0
        for req in new_load_list:
            current_buffer = self.cum_buffer_size[req.rid]
            req_len = len(req.origin_input_ids) + len(req.output_ids)

            total_value += value.get(req, 0) * (real_reschedule_interval - cum_load_time)
            estimate_buffer_size[req.rid] = current_buffer \
                    + (real_reschedule_interval - cum_load_time) / estimate_decode_time \
                    - real_reschedule_interval * self.output_speed[req.rid]
            cum_load_time += req_len / current_evicting_speed + req_len / current_loading_speed

        for req in new_prefill_list:
            current_buffer = self.cum_buffer_size[req.rid]
            total_value += value.get(req, 0) * (real_reschedule_interval - cum_load_time - estimate_prefill_time)
            estimate_buffer_size[req.rid] = current_buffer \
                    + (real_reschedule_interval - cum_load_time - estimate_prefill_time) / estimate_decode_time \
                    - real_reschedule_interval * self.output_speed[req.rid]

        for rid in self.cum_buffer_size:
            current_buffer = self.cum_buffer_size[rid]
            if rid not in estimate_buffer_size.keys():
                estimate_buffer_size[rid] = current_buffer \
                    - real_reschedule_interval * self.output_speed[rid]

        # Calculate the objective function term of buffer size
        buffer_value = 0
        for rid, buffer_size in estimate_buffer_size.items():
            if buffer_size < 0:
                buffer_value += -buffer_size

        alpha = 1.0
        beta = 0.001
        return alpha * total_value - beta * buffer_value

    def greedy_selection(self, schedule_decision, value, candidates=None):
        # Simple greedy algorithm for request selection
        if candidates is None:
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
                if self.tree_cache.can_load_back(req):
                    adjusted_value_load = v
                    candidates.append((req, adjusted_value_load, size, 'waiting', False))  # recompute=False
                # prefilling方式
                if self.next_prefill_batch is None or len(self.next_prefill_batch) == 0:
                    adjusted_value_recompute = v
                    candidates.append((req, adjusted_value_recompute, size, 'waiting', True))  # recompute=True

            # 按 adjusted value 排序，性价比高的优先
            candidates.sort(key=lambda x: -x[1])

        for req, adjusted_value, size, source, recompute in candidates:
            if source == 'running':
                if schedule_decision.can_add_request(req):
                    schedule_decision.add_request(req)
            else:  # waiting
                if schedule_decision.can_add_request(req, recompute=recompute):
                    schedule_decision.add_request(req, recompute=recompute)
    
    def local_search(self, schedule_decision, value):
        # After greedy search, we can use local search to do some small change to schedule decision
        improved = True

        while improved:
            improved = False
            current_objective = self.evaluate_objective(schedule_decision, value)

            all_reqs = schedule_decision.keep_running_list + \
                        schedule_decision.new_load_list + \
                        schedule_decision.new_prefill_list

            # 尝试删除一个，加一个
            for remove_req in all_reqs:
                for try_add_req in schedule_decision.running_reqs + schedule_decision.waiting_reqs:
                    if try_add_req in all_reqs:
                        continue  # 跳过已在的或loading中的
                    # 尝试移除remove_req
                    removed_recompute = remove_req in schedule_decision.new_prefill_list
                    schedule_decision.remove_request(remove_req)

                    # 尝试添加try_add_req
                    new_can_recompute = not self.tree_cache.can_load_back(try_add_req)
                    if schedule_decision.can_add_request(try_add_req, recompute=new_can_recompute):
                        if new_can_recompute and (self.next_prefill_batch is None or len(self.next_prefill_batch) == 0):
                            schedule_decision.add_request(try_add_req, recompute=new_can_recompute)
                            new_objective = self.evaluate_objective(schedule_decision, value)
                            if new_objective > current_objective:
                                # 接受更优结果
                                current_objective = new_objective
                                # print('successful remove', remove_req.rid)
                                improved = True
                                break
                            else:
                                # 恢复状态
                                schedule_decision.remove_request(try_add_req)
                                schedule_decision.add_request(remove_req, recompute=removed_recompute)
                        else:
                            # 恢复状态
                            schedule_decision.add_request(remove_req, recompute=removed_recompute)
                    else:
                        # 恢复状态
                        schedule_decision.add_request(remove_req, recompute=removed_recompute)
                if improved:
                    break
    
    def get_token_value(self):
        # currently we use e^{-x} as the valid throughput
        # e.g. if cum_buffer_size = 0, then throughput = 1
        # the larger the cum_buffer_size, the smaller the throughput
        alpha = 1.0
        beta = 0.000001
        scale = 1.0
        if self.rebuffer_time is not None and len(self.rebuffer_time) > 0:
            scale = max(self.rebuffer_time.values()) if max(self.rebuffer_time.values()) > 0 else 1.0

        v_tokens = {}
        if self.waiting_queue is not None:
            for req in self.waiting_queue:
                assert req.rid in self.cum_buffer_size, \
                    "Request not in cum_buffer_size, but it should be added when the request is received."
                # final_buffer_size = self.cum_buffer_size[req.rid] - \
                #     min(self.estimate_load_cost(req), self.estimate_recompute_cost(req)) * self.output_speed[req.rid]
                # # clamp the buffer size, avoid the overflow error
                # if final_buffer_size < -5: final_buffer_size = -5
                final_buffer_size = self.cum_buffer_size[req.rid]

                v_tokens[req] = alpha * math.exp(-final_buffer_size) + beta * self.rebuffer_time[req.rid] / scale
        
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                assert req.rid in self.cum_buffer_size, \
                    "Request not in cum_buffer_size, but it should be added when the request is received."
                v_tokens[req] = alpha * math.exp(-self.cum_buffer_size[req.rid]) + beta * self.rebuffer_time[req.rid] / scale
        
        return v_tokens
    
    def generate_schedule_decision(self):
        v_token = self.get_token_value()

        schedule_decision = MyScheduleDecision(self.token_to_kv_pool, self.cum_buffer_size, self.output_speed,
                                            [] if self.running_batch is None else self.running_batch.reqs, 
                                            self.waiting_queue, self.offload_manager.get_all_load_reqs(), self.offload_manager.get_all_evict_reqs(), 
                                            [] if self.next_prefill_batch is None else self.next_prefill_batch,
                                            self.max_running_requests, self.max_prefill_tokens, int(self.max_running_requests * 0.4), self.new_token_ratio)

        # Initalize the previous batch, and fill in the free slots
        schedule_decision.initialize_keep_running_list(self.token_to_kv_pool.available_size(), self.req_to_token_pool.available_size())
        # Get Approximate Valid throughput from v_token
        valid_thr = {}
        for req in v_token.keys():
            valid_thr[req] = v_token[req] * self.reschedule_interval
        self.greedy_selection(schedule_decision, valid_thr)

        if len(self.waiting_queue) > 0:
            running_queue_evict_candidate = []
            waiting_queue_run_candidate = []
            load_time = self.tree_cache.get_loading_workload()[0] / self.tree_cache.get_loading_workload()[1]
            write_time = self.tree_cache.get_writing_workload()[0] / self.tree_cache.get_writing_workload()[1]
            for req in schedule_decision.keep_running_list:
                if self.cum_buffer_size[req.rid] >= self.output_speed[req.rid] * (load_time + write_time + self.reschedule_interval):
                    running_queue_evict_candidate.append(req)
            for req in self.waiting_queue:
                # (req, req_len, adjusted_value, 'waiting', True)
                req_len = len(req.origin_input_ids) + len(req.output_ids) + 1
                if self.tree_cache.can_load_back(req):
                    adjust_value = v_token[req] * (self.reschedule_interval - req_len / self.tree_cache.get_loading_workload()[1] - req_len / self.tree_cache.get_writing_workload()[1])
                    waiting_queue_run_candidate.append((req, req_len, adjust_value, "waiting", False))
                if self.next_prefill_batch is None or len(self.next_prefill_batch) == 0:
                    adjust_value = v_token[req] * self.reschedule_interval
                    waiting_queue_run_candidate.append((req, req_len, adjust_value, "waiting", True))
            
            running_queue_evict_candidate = sorted(running_queue_evict_candidate, key=lambda x: (self.cum_buffer_size[x.rid], -self.output_speed[x.rid]))
            #evict_num = int(len(running_queue_evict_candidate))
            #running_queue_evict_candidate = running_queue_evict_candidate[-evict_num:]

            for req in running_queue_evict_candidate:
                schedule_decision.remove_request(req)
                req_len = len(req.origin_input_ids) + len(req.output_ids) + 64
                waiting_queue_run_candidate.append((req, req_len, valid_thr[req], "running", False))
            
            waiting_queue_run_candidate = [req for idx, req in enumerate(waiting_queue_run_candidate) \
                                           if self.cum_buffer_size[req[0].rid] <= 2.0 * self.output_speed[req[0].rid]]
            waiting_queue_run_candidate = sorted(waiting_queue_run_candidate, key=lambda x: (self.cum_buffer_size[x[0].rid], -self.output_speed[x[0].rid]))

            self.greedy_selection(schedule_decision, valid_thr, candidates=waiting_queue_run_candidate)
            self.local_search(schedule_decision, v_token)

        print('valid_thr', valid_thr, file=open('tmp/buffer_size.log', 'a'))
        for req in valid_thr.keys():
            print(f'({req.rid}, {self.estimate_load_cost(req)}, {req in schedule_decision.keep_running_list})', end=' ', file=open('tmp/buffer_size.log', 'a'))
        print('', file=open('tmp/buffer_size.log', 'a'))

        # DEBUG: Print the detail of schedule decision to log
        print('keep_running_list', schedule_decision.keep_running_list, file=open('tmp/schedule_output.txt', "a"))
        print(len(schedule_decision.keep_running_list), file=open('tmp/schedule_output.txt', "a"))
        print('new_load_list', schedule_decision.new_load_list, file=open('tmp/schedule_output.txt', "a"))
        print(len(schedule_decision.new_load_list), file=open('tmp/schedule_output.txt', "a"))
        print('new_prefill_list', schedule_decision.new_prefill_list, file=open('tmp/schedule_output.txt', "a"))
        print(len(schedule_decision.new_prefill_list), file=open('tmp/schedule_output.txt', "a"))
        print('waiting queue', self.waiting_queue, file=open('tmp/schedule_output.txt', "a"))
        print(len(self.waiting_queue), file=open('tmp/schedule_output.txt', "a"))
        print(self.token_to_kv_pool.available_size(), self.req_to_token_pool.available_size(), file=open('tmp/schedule_output.txt', "a"))
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

        # update_buffer_size()
        self.update_buffer_size()

        self.waiting_queue.extend(self.offload_manager.step())
        if self.next_prefill_batch is not None and len(self.next_prefill_batch) > 0:
            bs = len(self.next_prefill_batch)
            reqs = self.next_prefill_batch
            input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
            extend_num_tokens = sum(len(ids) for ids in input_ids)

            selected_reqs = []
            selected_bs = 0
            selected_tokens = 0
            for req in reqs:
                req_tokens = len(req.fill_ids[len(req.prefix_indices) :]) + 1
                if selected_tokens + req_tokens <= self.token_to_kv_pool.available_size() and \
                    selected_bs + 1 <= self.req_to_token_pool.available_size():
                    selected_reqs.append(req)
                    selected_bs += 1
                    selected_tokens += req_tokens

            if len(selected_reqs) > 0:
                ret = ScheduleBatch.init_new(
                    selected_reqs,
                    self.req_to_token_pool,
                    self.token_to_kv_pool,
                    self.tree_cache,
                    self.model_config,
                    self.enable_overlap,
                    self.spec_algorithm,
                    self.server_args.enable_custom_logit_processor,
                    self.server_args.return_hidden_states,
                )
                self.next_prefill_batch = [req for req in self.next_prefill_batch if req not in selected_reqs]
                if len(self.next_prefill_batch) == 0:
                    self.next_prefill_batch = None
                ret.prepare_for_extend()
                # print(f"prefill now!")
                # update request last run time
                for req in ret.reqs:
                    self.req_last_run_time[req.rid] = time.time()
                return ret
        
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

        if self.runtime_check and self.running_batch is not None:
            # print(self.running_batch)
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
    
        if (self.last_schedule is None or time.time() - self.last_schedule >= self.reschedule_interval):
            print("Write waiting:", self.tree_cache.write_token_num, self.tree_cache.wrote_token_num)
            
            self.last_schedule = time.time()
            if self.running_batch is not None:
                seq_lens_cpu = self.running_batch.seq_lens_cpu.numpy()
                before_req_nums = len(self.running_batch.reqs) + len(self.waiting_queue) + self.offload_manager.get_total_reqs()
            else:
                before_req_nums = len(self.waiting_queue) + self.offload_manager.get_total_reqs()
            print('cum_buffer_size', self.cum_buffer_size, file=open('tmp/buffer_size.log', 'a'))
            self.schedule_decision = self.generate_schedule_decision()
            
            if len(self.schedule_decision.new_prefill_list) == 0 and len(self.schedule_decision.keep_running_list) == 0 and len(self.schedule_decision.new_load_list) == 0:
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
                        # print(f'evict {req.rid} for {seq_lens_cpu[idx]} tokens')
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
                # print('running_batch len', len(self.running_batch.reqs))
                # print('waiting queue len', len(self.waiting_queue))
                self.running_batch.filter_batch(keep_indices=keep_indices)
                res = self.offload_manager.step()
                self.waiting_queue.extend(res)

                if self.running_batch.is_empty():
                    self.running_batch = None
            
            if len(self.schedule_decision.new_load_list) != 0:
                for req in self.schedule_decision.new_load_list:
                    if req not in self.waiting_queue:
                        assert False, f'{req} not in waiting queue'
                # first we filter the waiting queue
                self.waiting_queue = [
                    x for x in self.waiting_queue if x not in set(self.schedule_decision.new_load_list)
                ]
                # second we deal with the load process
                for i, req in enumerate(self.schedule_decision.new_load_list):
                    self.offload_manager.add_load_request(req)
                # print('req_to_token_pool', self.req_to_token_pool.available_size(), '/', self.req_to_token_pool.size)
                self.waiting_queue.extend(self.offload_manager.step())
            
            if len(self.schedule_decision.new_prefill_list) != 0:
                for req in self.schedule_decision.new_prefill_list:
                    if req not in self.waiting_queue:
                        assert False, f'{req} not in waiting queue'
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
                self.next_prefill_batch = self.schedule_decision.new_prefill_list

                bs = len(self.next_prefill_batch)
                reqs = self.next_prefill_batch
                input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
                extend_num_tokens = sum(len(ids) for ids in input_ids)
                if self.req_to_token_pool.available_size() >= bs and self.token_to_kv_pool.available_size() >= extend_num_tokens:
                    # print('directly do prefill after evict')
                    ret = ScheduleBatch.init_new(
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
                    self.next_prefill_batch = None
                    try:
                        # print(f"prefill now!")
                        ret.prepare_for_extend()
                    except Exception as e:
                        print(ret.reqs)
                        total_tokens = 0
                        for req in ret.reqs:
                            total_tokens += len(req.output_ids) + len(req.origin_input_ids)
                        print(total_tokens, self.token_to_kv_pool.available_size())
                        print(len(ret.reqs), self.req_to_token_pool.available_size())
                        print(len(self.schedule_decision.keep_running_list) + len(self.schedule_decision.new_load_list) + len(self.schedule_decision.new_prefill_list), 
                              self.req_to_token_pool.size)
                        raise e
                else:
                    if self.running_batch is None:
                        ret = None
                    else:
                        self.running_batch = self.update_running_batch(self.running_batch)
                        ret = self.running_batch

                # check while scheduling, no request is lost
                current_req_nums = self.offload_manager.get_total_reqs() + len(self.schedule_decision.new_prefill_list) + \
                    ((len(self.running_batch.reqs)) if (self.running_batch is not None) else 0) + \
                    ((len(self.waiting_queue)) if (self.waiting_queue is not None) else 0)
                try:
                    assert current_req_nums == before_req_nums, \
                        f"request number not match ({current_req_nums} {before_req_nums})"
                except Exception as e:
                    print(self.offload_manager.get_total_reqs(), len(self.schedule_decision.new_prefill_list), ((len(self.running_batch.reqs)) if (self.running_batch is not None) else 0), len(self.waiting_queue))
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
                    print(len(self.schedule_decision.new_prefill_list), ((len(self.running_batch.reqs)) if (self.running_batch is not None) else 0), self.waiting_queue)
                    raise e
                
                if self.running_batch is None:
                    ret = None
                else:
                    self.running_batch = self.update_running_batch(self.running_batch)
                    ret = self.running_batch
        else:
            if self.running_batch is None:
                ret = None
            else:
                try:
                    self.running_batch = self.update_running_batch(self.running_batch)
                except Exception as e:
                    print(self.running_batch.reqs)
                    raise e
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

        # update request last run time
        if ret is not None:
            for req in ret.reqs:
                self.req_last_run_time[req.rid] = time.time()
            
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
            print(f'Load: ', self.offload_manager.get_waiting_load(), file=open('tmp/debug_log.txt', 'a'))
            print(f'Evict: ', self.offload_manager.get_waiting_evict(), file=open('tmp/debug_log.txt', 'a'))
            print(f'Loading: ', self.offload_manager.get_ongoing_load(), file=open('tmp/debug_log.txt', 'a'))
            print(f'Evicting: ', self.offload_manager.get_ongoing_evict(), file=open('tmp/debug_log.txt', 'a'))
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
