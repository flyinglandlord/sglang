CUDA_VISIBLE_DEVICES=7 \
python3 -m sglang.launch_server --model-path /mtc/chenjunyi/models/llama3-8b \
    --port 8000 --disable-radix-cache --disable-overlap-schedule --trust-remote-code \
    --mem-fraction-static 0.5 --enable-scheduler custom --max-running-requests 150 # --enable-mixed-chunk 