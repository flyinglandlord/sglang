from sglang.srt.mem_cache.paged_allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    DoubleSparseTokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
    MHATokenToKVPoolHost,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController,
    CacheOperation,
)
import threading
import torch
import argparse
import logging
import time
import pandas as pd


class PseudoCache:
    def __init__(
        self,
        max_total_num_tokens,
        page_size,
        kv_cache_dtype,
        head_num,
        head_dim,
        layer_num,
        device,
        ratio,
        random_seq=False,
    ):
        self.token_to_kv_pool_device = MHATokenToKVPool(
            max_total_num_tokens,
            page_size=page_size,
            dtype=kv_cache_dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=False,
        )
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            max_total_num_tokens,
            dtype=kv_cache_dtype,
            device=device,
            kvcache=self.token_to_kv_pool_device,
        )
        self.token_to_kv_pool_host = MHATokenToKVPoolHost(
            self.token_to_kv_pool_allocator.get_kvcache(), ratio
        ) 
        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            self.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            load_cache_event=self.load_cache_event,
        )
        self.random_seq = random_seq
    
    def test_write(self, token_num):
        host_indices = self.token_to_kv_pool_host.alloc(token_num)
        if self.random_seq: # random combination
            device_indices = torch.randperm(self.token_to_kv_pool_device.size - 1, device=self.token_to_kv_pool_device.device)[:token_num]
        else:
            device_indices = torch.arange(token_num, device=self.token_to_kv_pool_device.device)
        if host_indices is None:
            return None, None
        # self.token_to_kv_pool_host.protect_write(host_indices)
        data = self.token_to_kv_pool_device.get_flat_data(device_indices)
        data_amount = data.nelement() * data.element_size()
        start_time = time.time()
        self.token_to_kv_pool_host.transfer(host_indices, data)
        end_time = time.time()
        # self.token_to_kv_pool_host.complete_io(host_indices)
        # self.token_to_kv_pool_host.update_backup(host_indices)
        return end_time - start_time, data_amount
    
    def test_load(self, token_num):
        if self.random_seq: # random combination
            host_indices = torch.randperm(self.token_to_kv_pool_host.size - 1, device=self.token_to_kv_pool_host.device)[:token_num]
        else:
            host_indices = torch.arange(token_num, device=self.token_to_kv_pool_host.device)
        device_indices = self.token_to_kv_pool_allocator.alloc(token_num)
        if device_indices is None:
            return None, None
        # self.token_to_kv_pool_host.protect_load(host_indices)
        data = self.token_to_kv_pool_host.get_flat_data(host_indices)
        data_amount = data.nelement() * data.element_size()
        start_time = time.time()
        self.token_to_kv_pool_device.transfer(device_indices, data)
        end_time = time.time()
        # self.token_to_kv_pool_host.complete_io(host_indices)
        # self.token_to_kv_pool_host.update_backup(host_indices)
        return end_time - start_time, data_amount
        

def main(args):
    if args.kv_dtype == "float32":
        kv_dtype = torch.float32
    elif args.kv_dtype == "float16":
        kv_dtype = torch.float16
    elif args.kv_dtype == "bfloat16":
        kv_dtype = torch.bfloat16
    elif args.kv_dtype == "fp8_e5m2":
        kv_dtype = torch.float8_e5m2
    elif args.kv_dtype == "fp8_e4m3":
        kv_dtype = torch.float8_e4m3
    elif args.kv_dtype == "int8":
        kv_dtype = torch.int8
    else:
        raise ValueError(f"Unsupported dtype: {args.kv_dtype}")
    cache = PseudoCache(
        args.size,
        args.page_size,
        kv_dtype,
        args.head_num,
        args.head_dim,
        args.layer_num,
        args.device,
        args.ratio,
        args.random_seq,
    )
    expr_name = f"size_{args.size}_kv_dtype_{args.kv_dtype}_randseq_{args.random_seq}"
    df_write = pd.DataFrame(columns=["token_num", "time", "data_amount", "throughput"])
    test_token_num = 8
    while test_token_num < args.size:
        # logging.info(f"Test write token num: {test_token_num}")
        write_time, write_data_amount = cache.test_write(test_token_num)
        write_data_amount /= 1024 ** 2
        # logging.info(f"Write time: {write_time * 1000:.2f} ms, data amount: {write_data_amount:.2f}, throughput: {write_data_amount / write_time:.2f} MB/s")
        df_write = pd.concat(
            [
                df_write,
                pd.DataFrame(
                    [[test_token_num, write_time, write_data_amount, write_data_amount / write_time]],
                    columns=["token_num", "time", "data_amount", "throughput"],
                ),
            ],
            ignore_index=True,
        )
        test_token_num *= 2
    df_write.to_csv(f"write_{expr_name}.csv", index=False)
    df_load = pd.DataFrame(columns=["token_num", "time", "data_amount", "throughput"])
    test_token_num = 8
    while test_token_num < args.size:
        # logging.info(f"Test load token num: {test_token_num}")
        load_time, load_data_amount = cache.test_load(test_token_num)
        if load_time is None:
            logging.info("Load failed")
            break
        load_data_amount /= 1024 ** 2
        # logging.info(f"Load time: {load_time * 1000:.2f} ms, data amount: {load_data_amount:.2f}, throughput: {load_data_amount / load_time:.2f} MB/s")
        df_load = pd.concat(
            [
                df_load,
                pd.DataFrame(
                    [[test_token_num, load_time, load_data_amount, load_data_amount / load_time]],
                    columns=["token_num", "time", "data_amount", "throughput"],
                ),
            ],
            ignore_index=True,
        )
        test_token_num *= 2
    df_load.to_csv(f"load_{expr_name}.csv", index=False)
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--page_size", type=int, default=1)
    parser.add_argument("--kv_dtype", type=str, default="float32")
    parser.add_argument("--head_num", type=int, default=12)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--layer_num", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ratio", type=float, default=2.0)
    parser.add_argument("--random_seq", action="store_true")
    args = parser.parse_args()
    main(args)
