CUDA_VISIBLE_DEVICES=7 \
python3 -m sglang.launch_server --model-path /mtc/chenjunyi/models/llama3-8b \
    --port 8000 --disable-radix-cache --disable-overlap-schedule --trust-remote-code --max-running-requests 25 \
    --mem-fraction-static 0.2 --enable-scheduler custom # --enable-mixed-chunk 