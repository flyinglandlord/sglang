PYTHONPATH=`pwd`/python python -m sglang.bench_serving  \
    --port 30000 \
    --backend sglang \
    --num-prompts 10 \
    --dataset-name sharegpt \
    --dataset-path /home/datasets/sharegpt_gpt4/ShareGPT_V3_unfiltered_cleaned_split.json \
    --sharegpt-context-len 4096 \
    --use-trace BurstGPT_1.csv --trace-scale 1e-4

mv ./benchmark_result.pkl ./pkls/$1.pkl
