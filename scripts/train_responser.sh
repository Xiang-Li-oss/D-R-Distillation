python dr_train.py \
    --dataset hotpotqa \
    --val_set dev \
    --model t5-base \
    --role solver \
    --batch_size 8 \
    --accumulate 2 \
    --devices 0 \
    --epoch 20

