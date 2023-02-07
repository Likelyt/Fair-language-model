export SUB_SAMPLE_SIZE=1000000
export GPU_ID='7'
export MODEL_SELECT='gpt2'

nohup python Classifier.py \
    --model_name_or_path=$MODEL_SELECT \
    --sub_sample_size=$SUB_SAMPLE_SIZE \
    --gpu_id=$GPU_ID \
    --train_batch_size=8 \
    --val_batch_size=8 \
    --block_size=256 \
> ../log/cls/res_09_03_{$MODEL_SELECT}_small_B_8_Sample_{$SUB_SAMPLE_SIZE}.txt &
