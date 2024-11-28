CUR_LOG_DIR="logs_sampling/api"

mkdir -p ${CUR_LOG_DIR}

srun -p INTERN2 \
    --gres=gpu:0 \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    --job-name "wwy_api" \
    -o "${CUR_LOG_DIR}/proxy.log" \
    -e "${CUR_LOG_DIR}/proxy.log" \
lmdeploy serve proxy \
    --server-name 0.0.0.0 \
    --server-port 8000 \
    --strategy min_observed_latency

