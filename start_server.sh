CUDA_VISIBLE_DEVICES=1,3,4,5,7 \
python3 -m sglang.launch_server --model-path /mnt/nvme0/models/Meta-Llama-3-8B \
    --port 8888 --disable-radix-cache --disable-overlap-schedule \
    --mem-fraction-static 0.3 --enable-scheduler custom # --enable-mixed-chunk 