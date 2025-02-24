CUDA_VISIBLE_DEVICES=0,1 \
python3 -m sglang.launch_server --model-path /mnt/nvme0/models/Meta-Llama-3-8B \
    --port 9999 --disable-radix-cache --enable-mixed-chunk \
    --disable-overlap-schedule --enable-custom-scheduler