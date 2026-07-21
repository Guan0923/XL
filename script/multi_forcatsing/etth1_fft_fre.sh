
model_name=XLinear-FFT-Fre
root_path_name=./dataset/
data_path_name=ETTh1.csv

data_name=ETTh1
random_seed=2025

log_dir=./logs
mkdir -p $log_dir

for pred_len in 96 192 336 720; do

case $pred_len in
    96)   threshold=40; channel_model=64; hidden_size=256; head_dropout=0.2; t_dropout=0; embed_dropout=0; d_model=128; t_ff=512;  patience=10; lr=0.001; batch_size=128 ;;
    192)  threshold=40; channel_model=64; hidden_size=256; head_dropout=0.3; t_dropout=0; embed_dropout=0; d_model=128; t_ff=1024; patience=5;  lr=0.001; batch_size=128 ;;
    336)  threshold=40; channel_model=96; hidden_size=256; head_dropout=0.4; t_dropout=0; embed_dropout=0; d_model=128; t_ff=1024; patience=5;  lr=0.001; batch_size=128 ;;
    720)  threshold=30; channel_model=128; hidden_size=128; head_dropout=0.2; t_dropout=0.1; embed_dropout=0; d_model=64;  t_ff=128;  patience=10; lr=0.001; batch_size=128 ;;
esac

model_id_name=ETTh1_${lr}

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_96_${pred_len} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --enc_in 7 \
    --d_model $d_model \
    --channel_model $channel_model \
    --hidden_size $hidden_size \
    --t_ff $t_ff \
    --t_dropout $t_dropout \
    --embed_dropout $embed_dropout \
    --head_dropout $head_dropout \
    --dropout 0.05 \
    --threshold $threshold \
    --patience $patience \
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size $batch_size \
    --itr 1 \
    --learning_rate $lr > ${log_dir}/${model_id_name}_${pred_len}.log 2>&1 &

done

wait
