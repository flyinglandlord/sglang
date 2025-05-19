# export SGLANG_TORCH_PROFILER_DIR=/mtc/chenjunyi/sglang/torch_profiler
CUDA_VISIBLE_DEVICES=1 \
python3 -m sglang.launch_server --model-path /mtc/chenjunyi/models/llama3-8b \
    --port 8000 --disable-radix-cache --disable-overlap-schedule --trust-remote-code \
    --enable-scheduler default --mem-fraction-static 0.3 # --chunked-prefill-size 256 # --schedule-conservativeness 0.0  # --enable-mixed-chunk 
    # --max-running-requests 150 --enable-scheduler custom # --schedule-conservativeness 0.0  # --enable-mixed-chunk 