PYTHONPATH=`pwd`/python python -m sglang.bench_serving  \
    --port 30000 \
    --backend sglang \
    --num-prompts 10 \
    --dataset-name sharegpt \
    --dataset-path trace.json \
    --sharegpt-context-len 4096 \
    --use-trace trace.log --trace-scale 7e-4

mv ./benchmark_result.pkl ./pkls/$1.pkl
