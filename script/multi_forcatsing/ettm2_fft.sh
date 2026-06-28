
model_name=XLinear-FFT
root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
random_seed=2025
log_dir=./logs

mkdir -p $log_dir

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTm2_96_96 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 7 \
    --d_model 512 \
    --head_dropout 0.6 \
    --dropout 0.05 \
    --top_k 5 \
    --threshold 50 \
    --patience 3 \
    --des 'Exp' \
    --train_epochs 20 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001 > ${log_dir}/${model_id_name}_96_fft.log 2>&1 &

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTm2_96_192 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --enc_in 7 \
    --d_model 512 \
    --head_dropout 0.7 \
    --dropout 0.05 \
    --top_k 5 \
    --threshold 50 \
    --des 'Exp' \
    --train_epochs 20 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001 > ${log_dir}/${model_id_name}_192_fft.log 2>&1 &

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTm2_96_336 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --enc_in 7 \
    --d_model 512 \
    --head_dropout 0.6 \
    --dropout 0.05 \
    --top_k 5 \
    --threshold 50 \
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001 > ${log_dir}/${model_id_name}_336_fft.log 2>&1 &

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTm2_96_720 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --enc_in 7 \
    --d_model 512 \
    --head_dropout 0.4 \
    --dropout 0.05 \
    --top_k 5 \
    --threshold 50 \
    --des 'Exp' \
    --train_epochs 15 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001 > ${log_dir}/${model_id_name}_720_fft.log 2>&1 &

wait
