model_name="sasrec"
dataset="fashion"

python main.py --dataset ${dataset} \
        --model_name ${model_name} \
        --inter_file "reverseTrain" \
        --pseudoNum 15 \
        --poolSize 20 \
        --candidatePool \
        --log