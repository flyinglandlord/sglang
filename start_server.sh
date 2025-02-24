CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
python3 -m sglang.launch_server --model-path /mnt/nvme0/models/Meta-Llama-3-8B \
    --port 9999 --disable-radix-cache --enable-mixed-chunk \
    --disable-overlap-schedule --enable-custom-scheduler