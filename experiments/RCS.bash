model_name="sasrec"
dataset="fashion"

python main.py --dataset ${dataset} \
        --model_name ${model_name} \
        --inter_file "inter"\
        --poolSize 10 \
        --aug_file "interAug" \
        --history_file "inter" \
        --ReliabilityTest \
        --log