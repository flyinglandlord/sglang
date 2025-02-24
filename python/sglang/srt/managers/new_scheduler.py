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
        self.log_batch_status = False
        self.virtual_buffer_size = {}
        self.decode_time_stamp = {}
        self.output_speed = 20
        self.avg_decode_time = 0.0
        self.avg_prefill_time = 0.0
        self.schedule_interval = 1.0

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
                
                for req in batch.reqs:
                    if req.rid not in self.decode_time_stamp:
                        self.decode_time_stamp[req.rid] = [time.time()]
                    else:
                        self.decode_time_stamp[req.rid].append(time.time())

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

    def get_next_batch_to_run(self):
        # First we process the prefill requests processed by the previous batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            # if the prefill request is chunked
            # we put them back to the waiting queue, cache the kv cache
            if self.being_chunked_req:
                # Move the chunked request out of the batch
                self.last_batch.filter_batch(being_chunked_req=self.being_chunked_req)
                self.tree_cache.cache_unfinished_req(self.being_chunked_req)
                # being chunked request keeps its rid but will get a new req_pool_idx
                self.req_to_token_pool.free(self.being_chunked_req.req_pool_idx)
                self.batch_is_full = False
            
            # otherwise, we merge them into the running batch
            # because they now become the decoding requests
            if not self.last_batch.is_empty():
                # update the virtual buffer size
                # primarily the virtual buffer size is 0, because there is no output token
                # for req in self.last_batch.reqs:
                #     self.virtual_buffer_size[req.rid] = 10

                # merge the last batch into the running batch
                if self.running_batch is None:
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)
        
        # Next we get the preference decode requests by their virtual buffer size
        # sort the decode request by the current virtual buffer size
        # current virtual buffer size = virtual buffer size - output speed * (current time - decode time stamp)
        if self.batch_is_full:
            # print(self.virtual_buffer_size)
            current_time = time.time()
            Q_wait = {}
            Q_service = {}
            for req in (self.running_batch.reqs + self.waiting_queue):
                current_output_len = len(req.output_ids)
                req_history_time = self.decode_time_stamp[req.rid]
                T_actual = req_history_time[0]
                T_ideal = req.recv_time + self.avg_prefill_time
                Q_history = max(0, T_actual[0] - T_ideal[0])
                # Calculate Q_history
                for i in range(1, current_output_len):
                    # T_actual
                    if T_actual >= req_history_time[i]:
                        T_actual = T_actual + 1 / self.output_speed
                    else:
                        T_actual = req_history_time[i]
                    # T_ideal
                    T_ideal = T_ideal + 1 / self.output_speed
                    # Q_history
                    Q_history += max(0, T_actual - T_ideal)
                
                # suppose we re-scheudle the request at the current time
                # next time we will schedule the request at the current time + 50 * self.avg_decode_time
                # Calculate Q_wait
                Q_service[req.rid] = Q_history
                Q_wait[req.rid] = Q_history
                wait_T_actual =  T_actual + self.schedule_interval
                service_T_actual = T_actual
                for i in range(self.schedule_interval // self.avg_decode_time):
                    Q_service[req.rid] += max(0, service_T_actual - T_ideal)
                    Q_wait[req.rid] += max(0, wait_T_actual - T_ideal)
                    wait_T_actual += 1 / self.output_speed
                    service_T_actual += 1 / self.output_speed

            # sort the request by the Q_wait - Q_service
            priority = [(req, Q_wait[req.rid] - Q_service[req.rid]) for req in self.waiting_queue]
            priority.sort(key=lambda x: x[1], reverse=True)

            # split the request by inner running_batch and outer running_batch
            running_priority = []
            waiting_priority = []
            for req, _ in priority:
                if req in self.running_batch.reqs:
                    running_priority.append(req)
                else:
                    waiting_priority.append(req)

            # check the request in the running_batch need to be swapped out
            # self.new_token_ratio
            # self.token_to_kv_pool.size
            # self.max_prefill_tokens
            max_tokens = self.token_to_kv_pool.size
            max_prefill_tokens = self.max_prefill_tokens
            new_token_ratio = self.new_token_ratio
            selected = []
            for req, pri in priority:
                if req in self.running_batch.reqs:
                    remain_tokens = max_tokens - self.running_batch.seq_lens[req.rid] - min(
                                (req.sampling_params.max_new_tokens - len(req.output_ids)),
                                CLIP_MAX_NEW_TOKENS_ESTIMATION,
                            ) * new_token_ratio
                    if remain_tokens >= 0:
                        max_tokens = remain_tokens
                        selected.append(req)
                    else:
                        continue
                else:
                    extend_len = len(req.fill_ids)
                    remain_tokens = max_tokens - extend_len - min(
                                (req.sampling_params.max_new_tokens - len(req.output_ids)),
                                CLIP_MAX_NEW_TOKENS_ESTIMATION,
                            ) * new_token_ratio
                    if remain_tokens >= 0 and max_prefill_tokens - extend_len >= 0:
                        max_tokens = remain_tokens
                        max_prefill_tokens -= extend_len
                        selected.append(req)
                    else:
                        continue

            # clear the request not in the selected list
            seq_lens_cpu = self.running_batch.seq_lens.cpu().numpy()
            for req in self.running_batch.reqs:
                if req not in selected:
                    if isinstance(self.tree_cache, ChunkCache):
                        # ChunkCache directly evict all tokens
                        token_indices = self.req_to_token_pool.req_to_token[
                            req.req_pool_idx, : seq_lens_cpu[req.rid]
                        ]
                        self.token_to_kv_pool.free(token_indices)
                        self.req_to_token_pool.free(req.req_pool_idx)
                        del self.tree_cache.entries[req.rid]
                    else:
                        assert False, "Only ChunkCache supports new scheduler"
                    
                    req.prefix_indices = []
                    req.last_node = None
                    req.extend_input_len = 0
                    req.is_retracted = True

                    # For incremental logprobs
                    req.last_update_decode_tokens = 0
                    req.logprob_start_len = 10**9

                    self.waiting_queue.append(req)
            
            # merge a new prefill batch in the running batch
            can_run_list = [r for r in waiting_priority if r in selected and r not in self.running_batch.reqs]
            new_batch = ScheduleBatch.init_new(
                can_run_list,
                self.req_to_token_pool,
                self.token_to_kv_pool,
                self.tree_cache,
                self.model_config,
                self.enable_overlap,
            )
            new_batch.prepare_for_extend()
            if (
                self.is_mixed_chunk
                and self.running_batch is not None
                and not (new_batch.return_logprob or self.running_batch.return_logprob)
            ):
                self.running_batch.filter_batch()
                if not self.running_batch.is_empty():
                    self.running_batch.prepare_for_decode()
                    new_batch.mix_with_running(self.running_batch)
                    new_batch.decoding_reqs = self.running_batch.reqs
                self.running_batch = None
            else:
                new_batch.decoding_reqs = None
            return new_batch
        
            # decode_reqs = []
            # for req in self.running_batch.reqs:
            #     self.virtual_buffer_size[req.rid] -= self.output_speed * (current_time - self.decode_time_stamp[req.rid])
            #     decode_reqs.append((req, self.virtual_buffer_size[req.rid]))
            # decode_reqs.sort(key=lambda x: x[1], reverse=True)

            # # filter out the evict_reqs from the decode_reqs, which satisfies the condition buffer_size >= 100
            # seq_lens_cpu = self.running_batch.seq_lens.cpu().numpy()
            # keep_indices = [i for i in range(len(self.running_batch.reqs))]
            # evict_reqs = []
            # for idx, _ in enumerate(self.running_batch.reqs):
            #     req = self.running_batch.reqs[idx]
            #     if self.virtual_buffer_size[req.rid] < 100:
            #         continue
            #     if isinstance(self.tree_cache, ChunkCache):
            #         # ChunkCache does not have eviction
                    
            #         token_indices = self.req_to_token_pool.req_to_token[
            #             req.req_pool_idx, : seq_lens_cpu[idx]
            #         ]
            #         self.token_to_kv_pool.free(token_indices)
            #         self.req_to_token_pool.free(req.req_pool_idx)
            #         del self.tree_cache.entries[req.rid]
            #     else:
            #         assert False, "Only ChunkCache supports new scheduler"

            #     req.prefix_indices = []
            #     req.last_node = None
            #     req.extend_input_len = 0
            #     req.is_retracted = True

            #     # For incremental logprobs
            #     req.last_update_decode_tokens = 0
            #     req.logprob_start_len = 10**9
            #     keep_indices.remove(idx)
            #     evict_reqs.append(req)

            # # remove the evict_reqs from the running batch
            # if len(evict_reqs) > 0:
            #     print(keep_indices)
            #     for i in evict_reqs:
            #         print(i.prefix_indices)
            #     self.running_batch.filter_batch(keep_indices=keep_indices)
            #     self.waiting_queue.extend(evict_reqs)

        # if the cumulated decode requests are enough, and all these decode requests have empty or small buffer size
        # we need to run the decode requests
        if self.running_batch is not None and len(self.running_batch.reqs) >= 32:
            self.running_batch = self.update_running_batch(self.running_batch)
            return self.running_batch

        # Run prefill first if possible
        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            return new_batch
        
        # Run decode
        if self.running_batch is None:
            return None
        self.running_batch = self.update_running_batch(self.running_batch)
        return self.running_batch


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
            self.running_batch,
            self.new_token_ratio,
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size(),
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
            res = adder.add_one_req(req)
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.batch_is_full = True
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
        if self.tp_rank == 0:
            self.log_prefill_stats(adder, can_run_list, running_bs, has_being_chunked)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
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
    
    def process_batch_result_decode(self, batch: ScheduleBatch, result):
        # update the virtual buffer size
        for req in batch.reqs:
            self.virtual_buffer_size[req.rid] += 1
        
        # do other things
        super().process_batch_result_decode(batch, result)


def run_adaptive_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    setproctitle.setproctitle("sglang::scheduler")

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    if dp_rank is None:
        configure_logger(server_args, prefix=f" TP{tp_rank}")
    else:
        configure_logger(server_args, prefix=f" DP{dp_rank} TP{tp_rank}")

    # set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    suppress_other_loggers()
    parent_process = psutil.Process().parent()

    try:
        scheduler = AdaptiveScheduler(server_args, port_args, gpu_id, tp_rank, dp_rank)
        pipe_writer.send(
            {"status": "ready", "max_total_num_tokens": scheduler.max_total_num_tokens}
        )
        if scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)