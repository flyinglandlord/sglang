/mnt/nvme0/chenjunyi/miniconda3/envs/llm-exp/bin/python -m sglang.bench_serving  \
    --port 8888 \
    --backend sglang \
    --num-prompts 200 \
    --dataset-name random \
    --dataset-path /mnt/nvme0/chenjunyi/project/sglang/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-input 4096 --random-output 2048 \
    --sharegpt-output-len 2048