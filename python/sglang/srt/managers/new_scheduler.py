import faulthandler
import logging
import os
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
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, get_bool_env_var, set_gpu_proc_affinity, suppress_other_loggers
from sglang.utils import get_exception_traceback
from sglang.srt.managers.scheduler import Scheduler
from sglang.global_config import global_config

logger = logging.getLogger(__name__)

class AdaptiveScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_batch_status = True
        self.virtual_buffer_size = {}
        self.decode_time_stamp = {}
        self.output_speed = 40
        self.avg_decode_time = 0.01
        self.avg_prefill_time = 0.1
        self.reschedule_interval = 1.0
        self.last_reschedule_time = None
    
    def predict_prefill_time(self, x, y, z):
        # Prefill time: coef [-9.42713569e-04  5.61623220e-06  2.46562667e-05] 
        # intercept_ 0.0034941820402902557
        # (x,y,z) -> (prefill_req_num, prefix_tokens, new_tokens)
        return -9.42713569e-04 * x + 5.61623220e-06 * y + \
            2.46562667e-05 * z + 0.0034941820402902557
    def predict_decode_time(self, x):
        # Decode time: [0.00015339] 
        # intercept_ 0.00816578707288733
        # (x) -> (decode_req_num)
        return 0.00015339 * x + 0.00816578707288733

    @torch.no_grad()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        time_stamp = 0.0
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            if self.server_args.enable_dp_attention:
                batch = self.prepare_dp_attn_batch(batch)

            self.cur_batch = batch

            if batch:
                if self.log_batch_status:
                    for req in batch.reqs:
                        if (batch.decoding_reqs is not None and req not in batch.decoding_reqs) or batch.decoding_reqs is None:
                            print(f'{req.rid}', end=' ', file=open('tmp/batch_detail.txt', 'a'))
                    print('', file=open('tmp/batch_detail.txt', 'a'))
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

    @torch.no_grad()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            if batch:
                result = self.run_batch(batch)
                result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # A dummy first batch to start the pipeline for overlap scheduler.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    self.process_batch_result(tmp_batch, None)

            if self.last_batch:
                tmp_batch, tmp_result = result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # Self-check and re-init some states when the server is idle
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    def is_time_to_reschedule(self):
        if not self.batch_is_full:
            return False
        if self.last_reschedule_time is None:
            self.last_reschedule_time = time.time()
            return False
        if time.time() - self.last_reschedule_time > self.reschedule_interval:
            self.last_reschedule_time = time.time()
            return True
        return False

    # Keep Old Code for QoE Calculation
    # current_output_len = len(req.output_ids)
    # req_history_time = self.decode_time_stamp[req.rid]
    # T_actual = req_history_time[0]
    # T_ideal = req.recv_time + self.avg_prefill_time
    # Q_history = T_actual - T_ideal
    # # Calculate Q_history
    # for i in range(1, current_output_len):
    #     # T_actual
    #     if T_actual >= req_history_time[i]:
    #         T_actual = T_actual + 1 / self.output_speed
    #     else:
    #         T_actual = req_history_time[i]
    #     # T_ideal
    #     T_ideal = T_ideal + 1 / self.output_speed
    #     # Q_history
    #     Q_history += T_actual - T_ideal
    
    # # suppose we re-scheudle the request at the current time
    # # next time we will schedule the request at the current time + 50 * self.avg_decode_time
    # # Calculate Q_wait
    # Q_service[req.rid] = Q_history
    # Q_wait[req.rid] = Q_history
    # wait_T_actual =  T_actual + self.reschedule_interval
    # service_T_actual = T_actual
    # for i in range(int(self.reschedule_interval // self.avg_decode_time)):
    #     Q_service[req.rid] += service_T_actual - T_ideal
    #     Q_wait[req.rid] += wait_T_actual - T_ideal
    #     wait_T_actual += 1 / self.output_speed
    #     service_T_actual += 1 / self.output_speed

    def predict_request_QoE(self, req, B, no_pred=False):
        # We have the recv_time(req.recv_time) and the output_speed(self.output_speed)
        # from self.decode_time_stamp get the list of decode output time
        # QoE = 1 - \sum_i (T_actual_i - T_ideal_i) / \sum_i (T_actual_n - T_ideal_i)
        # return Q_service(B), Q_wait
        current_time = time.time()
        recv_time = req.recv_time
        output_speed = self.output_speed
        T_output = self.decode_time_stamp[req.rid]
        output_len = len(T_output)
        assert output_len == len(T_output), "Decode time stamp length {} is not equal to output length {}".format(len(T_output), output_len)
        Q = None
        
        if output_len != 0:
            T_ideal = [recv_time + i / output_speed for i in range(output_len)]
            T_actual = [T_output[0]]

            # try to extend the T_actual, T_output and T_ideal in the schedule_interval
            if B != -1 and not no_pred:
                pred_decode_time = self.predict_decode_time(B)
                pred_decode_token = int(self.reschedule_interval / pred_decode_time)

                T_ideal = T_ideal + [T_ideal[-1] + i / output_speed for i in range(1, pred_decode_token)]
                T_output = T_output + [T_output[-1] + i * pred_decode_time for i in range(1, pred_decode_token)]
            elif not no_pred:
                T_ideal = T_ideal + [T_ideal[-1] + 1 / output_speed]
                T_output = T_output + [T_output[-1] + self.reschedule_interval + self.predict_decode_time(1)]
            
            for i in range(1, len(T_output)):
                if T_actual[-1] + 1 / output_speed >= T_output[i]:
                    T_actual.append(T_actual[-1] + 1 / output_speed)
                else:
                    T_actual.append(T_output[i])
        
            Q = 1 - sum([(T_actual[i] - T_ideal[i]) for i in range(len(T_output))]) \
                / sum([(T_actual[-1] - T_ideal[i]) for i in range(len(T_output))])
        else:
            assert no_pred == False, "The request has no output but we need to calculate the QoE"
            if B == -1:
                Q = 0
            else:
                Q = 1

        return Q
    
    def reschedule_requests_with_batch_size(self, B):
        schedulable = []
        Q_wait = {}
        Q_service = {}
        for req in self.running_batch.reqs:
            Q_service[req.rid] = self.predict_request_QoE(req, B)
            Q_wait[req.rid] = self.predict_request_QoE(req, -1)
            schedulable.append(req)
        for req in self.waiting_queue:
            Q_service[req.rid] = self.predict_request_QoE(req, B)
            Q_wait[req.rid] = self.predict_request_QoE(req, -1)
            schedulable.append(req)

        # sort the request by the (Q_wait - Q_service) / request_length
        priority = [(req, (Q_wait[req.rid] - Q_service[req.rid]) / (len(req.output_ids) + len(req.origin_input_ids)))  \
                    for req in schedulable]
        priority.sort(key=lambda x: x[1], reverse=True)

        # check the request in the running_batch need to be swapped out
        # self.new_token_ratio
        # self.token_to_kv_pool.size
        # self.max_prefill_tokens
        max_tokens = self.token_to_kv_pool.size
        max_running_requests = B
        max_prefill_tokens = self.max_prefill_tokens
        new_token_ratio = self.new_token_ratio
        selected = []
        total_tokens = 0
        new_tokens = 0
        seq_lens_cpu = self.running_batch.seq_lens.cpu().numpy()
        qoe_gain = 0
        for i, (req, pri) in enumerate(priority):
            if req in self.running_batch.reqs:
                idx = self.running_batch.reqs.index(req)
                remain_tokens = max_tokens - seq_lens_cpu[idx] - min(
                            (req.sampling_params.max_new_tokens - len(req.output_ids)),
                            CLIP_MAX_NEW_TOKENS_ESTIMATION,
                            self.reschedule_interval // self.predict_decode_time(B) + 3,
                        ) * new_token_ratio
                if remain_tokens >= 0 and max_running_requests >= 1:
                    max_tokens = remain_tokens
                    selected.append(req)
                    qoe_gain += (Q_wait[req.rid] - Q_service[req.rid])
                    total_tokens += seq_lens_cpu[idx] + min(
                            (req.sampling_params.max_new_tokens - len(req.output_ids)),
                            CLIP_MAX_NEW_TOKENS_ESTIMATION,
                        ) * new_token_ratio
                    max_running_requests -= 1
                else:
                    continue
            else:
                req.init_next_round_input(None)
                remain_tokens = max_tokens - req.extend_input_len - min(
                            (req.sampling_params.max_new_tokens - len(req.output_ids)),
                            CLIP_MAX_NEW_TOKENS_ESTIMATION,
                            self.reschedule_interval // self.predict_decode_time(B) + 3
                        ) * new_token_ratio
                if remain_tokens >= 0 and \
                    max_prefill_tokens - req.extend_input_len >= 0 and \
                    max_running_requests:
                    max_tokens = remain_tokens
                    max_prefill_tokens -= req.extend_input_len
                    selected.append(req)
                    qoe_gain += (Q_wait[req.rid] - Q_service[req.rid])
                    total_tokens += req.extend_input_len + min(
                            (req.sampling_params.max_new_tokens - len(req.output_ids)),
                            CLIP_MAX_NEW_TOKENS_ESTIMATION,
                        )
                    new_tokens += req.extend_input_len + min(
                            (req.sampling_params.max_new_tokens - len(req.output_ids)),
                            CLIP_MAX_NEW_TOKENS_ESTIMATION,
                        )
                    max_running_requests -= 1
                else:
                    continue
        return selected, qoe_gain, total_tokens, new_tokens

        
    def reschedule_requests(self):
        print('======Reschedule the requests======')
        # Try different batch size
        max_qoe_gain = -100000
        max_selected = None
        max_batch_size = -1
        max_total_tokens = 0
        max_new_tokens = 0
        
        # pre-determine the Bmin and Bmax
        Bmin = 1
        Bmax = self.max_running_requests
        # According to the paper, Bmin is set as the largest batch size that generates tokens 
        # faster than the most stringent user consumption speed across all requests.
        for B in range(1, self.max_running_requests):
            pred_decode_time = self.predict_decode_time(B)
            if pred_decode_time >= 1 / self.output_speed:
                Bmin = B
                break
        # Bmax is determined by adding to the batch requests with the shortest context 
        # lengths until the total number of tokens in the batch reaches M
        available_tokens = self.token_to_kv_pool.available_size()
        avg_context_length = 0
        avg_req_cnt = 0
        for req in self.waiting_queue:
            avg_context_length += len(req.origin_input_ids) + len(req.output_ids)
            avg_req_cnt += 1
        if avg_req_cnt != 0:
            avg_context_length /= avg_req_cnt
            Bmax = len(self.running_batch.reqs) + int(available_tokens / avg_context_length)
        else:
            Bmax = len(self.running_batch.reqs)

        logger.info(f"Predicted Bmin: {Bmin}, Bmax: {Bmax}")

        for B in range(min(Bmin-3, Bmax), Bmax+1):
            selected, qoe_gain, total_tokens, new_tokens = self.reschedule_requests_with_batch_size(B)
            # print(qoe_gain)
            if qoe_gain > max_qoe_gain:
                max_qoe_gain = qoe_gain
                max_selected = selected
                max_batch_size = len(selected)
                max_total_tokens = total_tokens
                max_new_tokens = new_tokens
        logger.info(f"Max QoE gain: {max_qoe_gain}, batch size: {max_batch_size}")
        self.last_reschedule_time = time.time()

        can_run_list = []
        for r in max_selected:
            if r not in self.running_batch.reqs:
                can_run_list.append(r)
        if len(can_run_list) == 0:
            return None
        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        # clear the request not in the selected list
        seq_lens_cpu = self.running_batch.seq_lens.cpu().numpy()
        swap_out = []
        for i, req in enumerate(self.running_batch.reqs):
            if req not in max_selected:
                if isinstance(self.tree_cache, ChunkCache):
                    # ChunkCache directly evict all tokens
                    token_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : seq_lens_cpu[i]
                    ]
                    self.token_to_kv_pool.free(token_indices)
                    self.req_to_token_pool.free(req.req_pool_idx)
                    if req.rid in self.tree_cache.entries:
                        del self.tree_cache.entries[req.rid]
                else:
                    assert False, "Only ChunkCache supports new scheduler"
                
                req.reset_for_retract()

                swap_out.append(req)
        self.running_batch.filter_batch(keep_indices=[i for i in range(len(self.running_batch.reqs)) if self.running_batch.reqs[i] in selected])
        self.waiting_queue.extend(swap_out)

        # Log the new_batch information
        logger.info("Re-schedule batch. #rescheduled-seq: %d. running-seq: %d. total-tokens: %d. new-tokens: %d" % 
                    (len(can_run_list), len(self.running_batch.reqs), max_total_tokens, max_new_tokens))
        
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
            self.server_args.return_hidden_states,
        )
        new_batch.prepare_for_extend()

        # Mixed-style chunked prefill
        # if (
        #     self.is_mixed_chunk
        #     and self.running_batch is not None
        #     and not (new_batch.return_logprob or self.running_batch.return_logprob)
        # ):
        #     # TODO (lianmin): support return_logprob + mixed chunked prefill
        #     self.running_batch.filter_batch()
        #     if not self.running_batch.is_empty():
        #         self.running_batch.prepare_for_decode()
        #         new_batch.mix_with_running(self.running_batch)
        #         new_batch.decoding_reqs = self.running_batch.reqs
        #     self.running_batch = None
        # else:
        #     new_batch.decoding_reqs = None
        
        return new_batch
    
    def get_next_batch_to_run(self):
        # Merge the prefill batch into the running batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
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

        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode
            if self.running_batch is None:
                ret = None
            elif self.is_time_to_reschedule():
                new_batch = self.reschedule_requests()
                if new_batch is None:
                    self.running_batch = self.update_running_batch(self.running_batch)
                    ret = self.running_batch
                else:
                    ret = new_batch
            else:
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch

        # Handle DP attention
        if self.server_args.enable_dp_attention:
            ret = self.prepare_dp_attn_batch(ret)

        return ret

    def get_new_batch_prefill(self):
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # Handle the cases where prefill is not allowed
        if (
            self.batch_is_full or len(self.waiting_queue) == 0
        ) and self.being_chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs) if self.running_batch else 0
        if running_bs >= self.max_running_requests:
            self.batch_is_full = True
            return None

        # Get priority queue
        prefix_computed = self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
        adder = PrefillAdder(
            self.tree_cache,
            self.token_to_kv_pool,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
        )

        has_being_chunked = self.being_chunked_req is not None
        if has_being_chunked:
            self.being_chunked_req.init_next_round_input()
            self.being_chunked_req = adder.add_being_chunked_req(self.being_chunked_req)

        if self.lora_paths:
            lora_set = (
                set([req.lora_path for req in self.running_batch.reqs])
                if self.running_batch is not None
                else set([])
            )

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if (
                self.lora_paths
                and len(
                    lora_set
                    | set([req.lora_path for req in adder.can_run_list])
                    | set([req.lora_path])
                )
                > self.max_loras_per_batch
            ):
                self.batch_is_full = True
                break

            if running_bs + len(adder.can_run_list) >= self.max_running_requests:
                self.batch_is_full = True
                break

            req.init_next_round_input(None if prefix_computed else self.tree_cache)

            if self.enable_hierarchical_cache and req.last_node is not None:
                if req.last_node.evicted:
                    # loading KV cache for the request
                    req.last_node, req.prefix_indices = self.tree_cache.init_load_back(
                        req.last_node,
                        req.prefix_indices,
                        adder.rem_total_tokens,
                    )
                    if req.last_node.loading:
                        # to prevent frequent cache invalidation
                        if req.rid in self.staging_reqs:
                            self.tree_cache.dec_lock_ref(self.staging_reqs[req.rid])
                        self.tree_cache.inc_lock_ref(req.last_node)
                        self.staging_reqs[req.rid] = req.last_node
                        continue
                elif req.last_node.loading:
                    if not self.tree_cache.loading_complete(req.last_node):
                        continue

                if req.rid in self.staging_reqs:
                    self.tree_cache.dec_lock_ref(self.staging_reqs[req.rid])
                    del self.staging_reqs[req.rid]

            res = adder.add_one_req(req)
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.batch_is_full = len(adder.can_run_list) > 0 or (
                            self.running_batch is not None
                            and not self.running_batch.is_empty()
                        )
                    else:
                        self.batch_is_full = True
                break
            if self.server_args.prefill_only_one_req:
                break

        # Update waiting queue
        can_run_list = adder.can_run_list
        if len(can_run_list) == 0:
            return None
        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if adder.new_being_chunked_req is not None:
            assert self.being_chunked_req is None
            self.being_chunked_req = adder.new_being_chunked_req

        if self.being_chunked_req:
            self.being_chunked_req.is_being_chunked += 1

        # Print stats
        if self.attn_tp_rank == 0:
            self.log_prefill_stats(adder, can_run_list, running_bs, has_being_chunked)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
            self.server_args.return_hidden_states,
        )
        new_batch.prepare_for_extend()

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and self.running_batch is not None
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = None
        else:
            new_batch.decoding_reqs = None

        return new_batch
    
    def calc_qoe(self, req):
        return self.predict_request_QoE(req, -1, no_pred=True)
    
    def process_batch_result_decode(self, batch: ScheduleBatch, result):
        # update the virtual buffer size & decode time stamp
        #for req in batch.reqs:
        #    self.virtual_buffer_size[req.rid] += 1
        for req in batch.reqs:
            if req.rid not in self.decode_time_stamp:
                self.decode_time_stamp[req.rid] = [time.time()]
            else:
                self.decode_time_stamp[req.rid].append(time.time())
            
            req.check_finished()
            if req.finished():
                service_qoe = self.calc_qoe(req)
                print(f"{req.rid}, {service_qoe}", file=open('tmp/service_qoe.txt', 'a'))
        
        # do other things
        super().process_batch_result_decode(batch, result)

    def process_batch_result_prefill(self, batch, result):
        for req in batch.reqs:
            if req.rid not in self.decode_time_stamp:
                self.decode_time_stamp[req.rid] = [time.time()]
            else:
                self.decode_time_stamp[req.rid].append(time.time())

            req.check_finished()
            if req.finished():
                service_qoe = self.calc_qoe(req)
                print(f"{req.rid}, {service_qoe}", file=open('tmp/service_qoe.txt', 'a'))
        return super().process_batch_result_prefill(batch, result)


def run_adaptive_scheduler_process(
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
        scheduler = AdaptiveScheduler(server_args, port_args, gpu_id, tp_rank, dp_rank)
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