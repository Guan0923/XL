
model_name=XLinear-FFT-X
root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
random_seed=2025

log_dir=./logs
mkdir -p $log_dir

for pred_len in 96 192 336 720; do

case $pred_len in
    96)   thresholds="40"; d_model=128; t_ff=512;   t_dropout=0;  c_ff=7;   c_dropout=0;  embed_dropout=0;  head_dropout=0.2; patience=10; lr=0.0005; batch_size=128 ;;
    192)  thresholds="40"; d_model=128; t_ff=1024;  t_dropout=0;  c_ff=7;   c_dropout=0;  embed_dropout=0;  head_dropout=0.3; patience=5;  lr=0.0005; batch_size=128 ;;
    336)  thresholds="40"; d_model=128; t_ff=1024;  t_dropout=0;  c_ff=7;   c_dropout=0;  embed_dropout=0;  head_dropout=0.4; patience=5;  lr=0.0005; batch_size=128 ;;
    720)  thresholds="20"; d_model=64;  t_ff=128;   t_dropout=0.1; c_ff=7;  c_dropout=0;  embed_dropout=0;  head_dropout=0.2; patience=10; lr=0.0005; batch_size=128 ;;
esac

for threshold in $thresholds; do

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_96_${pred_len}_th${threshold} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --enc_in 7 \
    --d_model $d_model \
    --t_ff $t_ff \
    --t_dropout $t_dropout \
    --c_ff $c_ff \
    --c_dropout $c_dropout \
    --embed_dropout $embed_dropout \
    --head_dropout $head_dropout \
    --dropout 0.05 \
    --threshold $threshold \
    --patience $patience \
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size $batch_size \
    --itr 1 \
    --learning_rate $lr > ${log_dir}/${model_id_name}_${pred_len}_th${threshold}.log 2>&1 &

done
done

wait
