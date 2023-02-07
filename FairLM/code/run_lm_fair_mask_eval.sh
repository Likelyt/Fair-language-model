export TRAIN_FILE=../data/wikitext-103-raw/wili.train.raw
export TEST_FILE=../data/wikitext-103-raw/wiki.test.raw
export OUTPUT_DIR=../output/pretrained/lm/block-256/regular/checkpoint-200000
export MASK=False
export GPU_BATCH_SIZE=8
export MODEL_TYPE=gpt2
export SUB_SAMPLE=10
export EPOCHS=4

nohup python lm_finetuning.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$MODEL_TYPE \
    --model_name_or_path=gpt2 \
    --block_size=256 \
    --sub_sample=$SUB_SAMPLE \
    --mask=$MASK \
    --mask_fair=$MASK \
    --mlm \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs=$EPOCHS \
    --per_gpu_train_batch_size=$GPU_BATCH_SIZE \
    --per_gpu_eval_batch_size=$GPU_BATCH_SIZE \
    --save_steps=100000 \
    --overwrite_output_dir \
> ../log/lm/no_fair_mask/res_{$MODEL_TYPE}_256_Batch_{$GPU_BATCH_SIZE}_mask_{$MASK}_Sample_{$SUB_SAMPLE}_epoch_{$EPOCHS}_evalstep_200000.txt &