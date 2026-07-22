model_name=XLinear
model_tag=xlinear
root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
random_seed=2025

train_epochs=100
patience=20
learning_rate=0.0005
baseline_lradj=type3
lgf_eta_min=1e-7
lgf_eta_max=$learning_rate
lgf_plateau_factor=0.8

log_dir=./logs
run_id="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$log_dir"

run_experiment() {
    local pred_len=$1
    local d_model=$2
    local t_ff=$3
    local c_ff=$4
    local c_dropout=$5
    local t_dropout=$6
    local head_dropout=$7
    local mode=$8
    local adaptive_args=()

    if [[ "$mode" == "ALGRS_plateau" ]]; then
        adaptive_args=(
            --use_lgflr 1
            --lgf_mode plateau
            --lgf_plateau_factor "$lgf_plateau_factor"
            --lgf_eta_min "$lgf_eta_min"
            --lgf_eta_max "$lgf_eta_max"
        )
    fi

    python -u run_longExp.py \
        --random_seed "$random_seed" \
        --is_training 1 \
        --root_path "$root_path_name" \
        --data_path "$data_path_name" \
        --model_id "${model_id_name}_96_${pred_len}" \
        --model "$model_name" \
        --data "$data_name" \
        --features M \
        --seq_len 96 \
        --pred_len "$pred_len" \
        --enc_in 7 \
        --d_model "$d_model" \
        --t_ff "$t_ff" \
        --c_ff "$c_ff" \
        --c_dropout "$c_dropout" \
        --t_dropout "$t_dropout" \
        --head_dropout "$head_dropout" \
        --embed_dropout 0 \
        --patience "$patience" \
        --des "$mode" \
        --train_epochs "$train_epochs" \
        --batch_size 128 \
        --itr 1 \
        --lradj "$baseline_lradj" \
        --learning_rate "$learning_rate" \
        "${adaptive_args[@]}" \
        > "${log_dir}/${model_id_name}_96_${pred_len}_${model_tag}_${mode}_${run_id}.log" 2>&1
}

run_suite() {
    local mode=$1
    echo "Starting ${model_name} ${mode} ablation"
    run_experiment 96 128 512 7 0 0 0.2 "$mode" &
    run_experiment 192 128 1024 7 0 0 0.3 "$mode" &
    run_experiment 336 128 1024 7 0 0 0.4 "$mode" &
    run_experiment 720 64 128 7 0 0.1 0.2 "$mode" &
    wait
    echo "Finished ${model_name} ${mode} ablation"
}

run_suite ALGRS_plateau
run_suite baseline
