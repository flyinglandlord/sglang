CUDA_VISIBLE_DEVICES=0 \
python3 -m sglang.launch_server --model-path /mtc/chenjunyi/models/llama3-8b \
    --port 8888 --disable-radix-cache --disable-overlap-schedule --trust-remote-code \
    --mem-fraction-static 0.3 --enable-scheduler custom # --enable-mixed-chunk 