rm /mtc/chenjunyi/sglang/tmp/batch_info.txt
rm /mtc/chenjunyi/sglang/tmp/batch_detail.txt
rm /mtc/chenjunyi/sglang/tmp/debug_log.txt
rm /mtc/chenjunyi/sglang/tmp/mem_log.log
rm /mtc/chenjunyi/sglang/tmp/offload_log.log
rm /mtc/chenjunyi/sglang/tmp/output_speed_info.log
rm /mtc/chenjunyi/sglang/tmp/schedule_output.txt
rm /mtc/chenjunyi/sglang/tmp/buffer_size.log

# export SGLANG_TORCH_PROFILER_DIR=/mtc/chenjunyi/sglang/torch_profiler

/mtc/yongyang/miniconda/envs/sgl_test/bin/python -m sglang.bench_serving  \
    --port 8000 \
    --backend sglang \
    --num-prompts 200 \
    --dataset-name random \
    --dataset-path /home/devsft/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-input 2048 --random-output 512 \
    --sharegpt-output-len 2048 # --profile # --request-rate 10 \