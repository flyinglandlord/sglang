rm /mnt/nvme0/chenjunyi/project/sglang/tmp/batch_info.txt
rm /mnt/nvme0/chenjunyi/project/sglang/tmp/batch_detail.txt
rm /mnt/nvme0/chenjunyi/project/sglang/tmp/debug_log.txt

/mnt/nvme0/chenjunyi/miniconda3/envs/llm-exp/bin/python -m sglang.bench_serving  \
    --port 8888 \
    --backend sglang \
    --num-prompts 75 \
    --dataset-name random \
    --dataset-path /mnt/nvme0/chenjunyi/project/sglang/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-input 4096 --random-output 20 \
    --sharegpt-output-len 2048