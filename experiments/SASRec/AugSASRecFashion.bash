gpu_id=3
dataset="fashion"
seed_list=(42 43 44)
ts_user=14
ts_item=4

model_name="aug_sasrec"
for seed in ${seed_list[@]}
do
        python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --seqWeight_file "weightDecay0.5"\
                --history_file "inter" \
                --aug_file "interAug" \
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
                --use_aug \
                --log
done
