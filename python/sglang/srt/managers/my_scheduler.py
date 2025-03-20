import faulthandler
import logging
import math
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
from python.sglang.srt.mem_cache.hichunk_cache import HiChunkCache
from python.sglang.srt.mem_cache.hiradix_cache import HiRadixCache
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
        self.log_batch_status = True
        self.output_speed = 25.0
        self.decode_time_stamp = {}
        self.cum_buffer_size = {}
        self.rebuffer_time = {}
        self.last_schedule = None
        self.reschedule = False
        self.last_update_buffer = None
        self.high_watermark = self.output_speed * 5.0
        self.low_watermark = self.output_speed * 1.0

        # We force the scheduler to use CPU-GPU Radix Cache
        self.tree_cache = ChunkCache(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
        )
    
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
        return super().handle_generate_request(recv_req)

    def update_buffer_size(self):
        # update cum_buffer_size and rebuffer time
        if self.last_update_buffer is None:
            self.last_update_buffer = time.time()
        else:
            during_time = time.time() - self.last_update_buffer
            for rid in self.cum_buffer_size:
                # print(f"{rid} {self.cum_buffer_size[rid]} {during_time} {self.output_speed}")
                if self.cum_buffer_size[rid] - during_time * self.output_speed >= 0:
                    self.cum_buffer_size[rid] -= during_time * self.output_speed
                else:
                    self.rebuffer_time[rid] += during_time - \
                        self.cum_buffer_size[rid] / self.output_speed
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
        if self.last_batch and self.last_batch.forward_mode.is_extend() and self.reschedule:
            print('trigger here')
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

        self.update_buffer_size()

        if self.last_schedule is None or time.time() - self.last_schedule >= 1.0:
            self.last_schedule = time.time()
            vthrouput = self.get_valid_throughput()
            # sort the request, by vthroughput and rebuffer time from high to low
            sorted_req = sorted(vthrouput.keys(), key=lambda x: vthrouput[x], reverse=True)
            before_req_nums = len(sorted_req)
            new_prefill_list = []
            keep_decode_list = []
            max_tokens = self.token_to_kv_pool.size
            # print("max_tokens", max_tokens)
            max_running_requests = self.max_running_requests
            max_prefill_tokens = self.max_prefill_tokens
            new_token_ratio = self.new_token_ratio
            if self.running_batch is not None:
                seq_lens_cpu = self.running_batch.seq_lens.cpu().numpy()
            for req in sorted_req:
                if self.running_batch is not None and req in self.running_batch.reqs:
                    idx = self.running_batch.reqs.index(req)
                    remain_tokens = max_tokens - seq_lens_cpu[idx] - min(
                                (req.sampling_params.max_new_tokens - len(req.output_ids))* new_token_ratio,
                                max(self.high_watermark - self.cum_buffer_size[req.rid], 0),
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
                    remain_tokens = max_tokens - req.extend_input_len - min(
                                (req.sampling_params.max_new_tokens - len(req.output_ids))* new_token_ratio,
                                max(self.high_watermark - self.cum_buffer_size[req.rid], 0),
                            ) 
                    if remain_tokens >= 0 and \
                    max_prefill_tokens - req.extend_input_len >= 0 and \
                    max_running_requests:
                        # print("selected", req.rid, max_tokens-remain_tokens, req.extend_input_len)
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
            
            swap_out = []
            # print(self.token_to_kv_pool.available_size())
            # print(len(new_prefill_list), len(keep_decode_list), 
            #       len(self.running_batch.reqs) if self.running_batch is not None else 0, len(self.waiting_queue))
            # first sweep out the running batch request not in keep_decode_list
            if self.running_batch is not None:
                before_running_batch = len(self.running_batch.reqs)
                for i, req in enumerate(self.running_batch.reqs):
                    if req not in keep_decode_list:
                        if isinstance(self.tree_cache, ChunkCache):
                            # ChunkCache directly evict all tokens
                            token_indices = self.req_to_token_pool.req_to_token[
                                req.req_pool_idx, : seq_lens_cpu[i]
                            ]
                            print(seq_lens_cpu[i])
                            self.token_to_kv_pool.free(token_indices)
                            self.req_to_token_pool.free(req.req_pool_idx)
                            if req.rid in self.tree_cache.entries:
                                del self.tree_cache.entries[req.rid]
                        else:
                            assert False, "Only ChunkCache supports new scheduler"
                        req.reset_for_retract()
                        swap_out.append(req)
                keep_indices = []
                for req in keep_decode_list:
                    keep_indices.append(self.running_batch.reqs.index(req))
                self.running_batch.filter_batch(keep_indices=keep_indices)
                assert len(self.running_batch.reqs) == len(keep_indices), \
                    f"fucking filter batch function not working {len(self.running_batch.reqs)} {len(keep_indices)}"
                self.waiting_queue.extend(swap_out)
                after_running_batch = len(self.running_batch.reqs)
                try:
                    assert before_running_batch == after_running_batch + len(swap_out), \
                        f"running batch size not match {before_running_batch} {after_running_batch} {len(swap_out)}"
                except Exception as e:
                    print(keep_decode_list, swap_out, keep_indices)
                    raise e
            
            if len(new_prefill_list) != 0:
                self.waiting_queue = [
                    x for x in self.waiting_queue if x not in set(new_prefill_list)
                ]
                total_new_token = 0
                for i in new_prefill_list:
                    total_new_token += i.extend_input_len
                try:
                    assert total_new_token <= self.token_to_kv_pool.available_size(), \
                        f"new token {total_new_token} exceed the available size {self.token_to_kv_pool.available_size()}"
                except Exception as e:
                    raise e
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
                current_req_nums = len(new_prefill_list) + \
                    ((len(self.running_batch.reqs)) if (self.running_batch is not None) else 0) + \
                    ((len(self.waiting_queue)) if (self.waiting_queue is not None) else 0)
                try:
                    assert current_req_nums == before_req_nums, \
                        f"request number not match ({current_req_nums} {before_req_nums})"
                except Exception as e:
                    print(len(new_prefill_list), len(self.running_batch.reqs), len(self.waiting_queue))
                    raise e
                self.reschedule = True
            else:
                current_req_nums = \
                    ((len(self.running_batch.reqs)) if (self.running_batch is not None) else 0) + \
                    ((len(self.waiting_queue)) if (self.waiting_queue is not None) else 0)
                try:
                    assert current_req_nums == before_req_nums, \
                        f"request number not match ({current_req_nums} {before_req_nums})"
                except Exception as e:
                    print(len(new_prefill_list), self.running_batch, self.waiting_queue)
                    raise e
                self.running_batch = self.update_running_batch(self.running_batch)
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