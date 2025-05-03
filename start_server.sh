export SGLANG_TORCH_PROFILER_DIR=/mtc/chenjunyi/sglang/torch_profiler
CUDA_VISIBLE_DEVICES=7 \
python3 -m sglang.launch_server --model-path /mtc/chenjunyi/models/llama3-8b \
    --port 8000 --disable-radix-cache --disable-overlap-schedule --trust-remote-code \
    --mem-fraction-static 0.5 --enable-scheduler custom --max-running-requests 50 # --schedule-conservativeness 0.0  # --enable-mixed-chunk 