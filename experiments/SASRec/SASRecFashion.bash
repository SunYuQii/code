gpu_id=0
dataset="fashion"
seed_list=(42 43 44)
ts_user=3
ts_item=4

model_name="sasrec"
for seed in ${seed_list[@]}
do
        python main.py --dataset ${dataset} \
                --inter_file "inter"\
                --model_name ${model_name} \
                --hidden_size 64 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --log
done