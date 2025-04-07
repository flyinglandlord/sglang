rm /mtc/chenjunyi/sglang/tmp/batch_info.txt
rm /mtc/chenjunyi/sglang/tmp/batch_detail.txt
rm /mtc/chenjunyi/sglang/tmp/debug_log.txt
rm /mtc/chenjunyi/sglang/tmp/mem_log.log

/mtc/yongyang/miniconda/envs/sgl_test/bin/python -m sglang.bench_serving  \
    --port 8888 \
    --backend sglang \
    --num-prompts 150 \
    --dataset-name random \
    --dataset-path /home/devsft/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-input 4096 --random-output 1024 \
    --sharegpt-output-len 2048