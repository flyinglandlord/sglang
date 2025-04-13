import torch
from typing import Dict, List


ENABLE_QUERY_COLLECTOR = False # or ENABLE_KV_SELECTOR

class SingletonMeta(type):
    _instances: Dict[type, object] = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class QueryCollector(metaclass=SingletonMeta):
    def __init__(self):
        self.queries: List[torch.Tensor] = []
    
    def add_query(self, query: torch.Tensor) -> None:
        # NOTE: (1) This operation is not permitted when using CUDA graph
        # (2) The performance of this operation is not guaranteed to be good
        # (3) Under tensor parallelism or data parallelism, the query tensor
        #     may be split into multiple tensors
        self.queries.append(query.detach().cpu())

    def reset(self) -> None:
        self.queries.clear()
    
    def fetch_all(self) -> torch.Tensor:
        assert len(self.queries) > 0, "No queries to fetch"
        queries = torch.stack(self.queries, dim=1)
        self.queries.clear()
        # print(f"query shape {queries.shape}")
        return queries # (batch_size, layer_num, head_size * hidden_size)
