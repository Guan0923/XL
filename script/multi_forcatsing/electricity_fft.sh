
model_name=XLinear-FFT
root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
random_seed=2025

log_dir=./logs
mkdir -p $log_dir

for top_k in 3 5 7; do
for threshold in 20 30 40; do

for pred_len in 96 192 336 720; do

case $pred_len in
    96)  head_dropout=0.2; patience=3; lr=0.0002; d_model=512; batch_size=16 ;;
    192) head_dropout=0.3; patience=5; lr=0.0001; d_model=512; batch_size=16 ;;
    336) head_dropout=0.3; patience=5; lr=0.0001; d_model=512; batch_size=16 ;;
    720) head_dropout=0.6; patience=5; lr=0.0002; d_model=512; batch_size=32 ;;
esac

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_96_${pred_len}_tk${top_k}_th${threshold} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --enc_in 321 \
    --d_model $d_model \
    --head_dropout $head_dropout \
    --dropout 0.05 \
    --top_k $top_k \
    --threshold $threshold \
    --patience $patience \
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size $batch_size \
    --itr 1 \
    --learning_rate $lr > ${log_dir}/${model_id_name}_${pred_len}_tk${top_k}_th${threshold}.log 2>&1

done
done
done
