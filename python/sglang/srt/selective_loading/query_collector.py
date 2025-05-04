import torch
from typing import Dict, List, Optional, Tuple


ENABLE_QUERY_COLLECTOR = True # or ENABLE_KV_SELECTOR
SAMPLED_LAYERS = [0]

class SingletonMeta(type):
    _instances: Dict[type, object] = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class QueryCollector(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.queries: torch.Tensor = None
        self.cur_layer: int = 0
        self.added_layer_num: int = 0
        self.layer_num: int = 0
        self.sampled_layers = set(SAMPLED_LAYERS)
    
    def init_query_collector(self, max_token_num: int, hidden_dim: int, layer_num: int, dtype: torch.dtype):
        print(f"QueryCollector: max_token_num={max_token_num}, hidden_dim={hidden_dim}, layer_num={layer_num}")
        self.queries = torch.zeros(
            (len(self.sampled_layers), max_token_num, hidden_dim), 
            dtype=dtype, pin_memory=True, device='cpu',
        )
        self.layer_num = layer_num
    
    def add_query(self, query: torch.Tensor) -> None:
        # NOTE: (1) This operation is not permitted when using CUDA graph
        # (2) The performance of this operation is not guaranteed to be good
        # (3) Under tensor parallelism or data parallelism, the query tensor
        #     may be split into multiple tensors
        if self.cur_layer in self.sampled_layers:
            self.queries[self.added_layer_num, :query.shape[0]].copy_(query, non_blocking=True)
            self.added_layer_num += 1
        self.cur_layer += 1
        if self.cur_layer == self.layer_num:
            self.cur_layer = 0
            self.added_layer_num = 0

    def reset(self) -> None:
        self.cur_layer = 0
        self.added_layer_num = 0
    
    def fetch_all(self, token_num: int) -> torch.Tensor:
        return self.queries[:, :token_num].clone(memory_format=torch.contiguous_format)
