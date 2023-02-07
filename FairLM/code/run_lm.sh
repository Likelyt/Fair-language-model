export TRAIN_FILE=../data/wikitext-103-raw/wili.train.raw
export TEST_FILE=../data/wikitext-103-raw/wiki.test.raw
export OUTPUT_DIR=../output/pretrained/lm/block-256/regular
export MODEL_TYPE=gpt2
export GPU_BATCH_SIZE=8
export SUB_SAMPLE=10
export EPOCHS=4
export MASK_FAIR=False

nohup python lm_finetuning.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$MODEL_TYPE \
    --model_name_or_path=$MODEL_TYPE \
    --block_size=256 \
    --train_data_file=$TRAIN_FILE \
    --sub_sample=$SUB_SAMPLE \
    --do_train \
    --mlm \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs=$EPOCHS \
    --per_gpu_train_batch_size=$GPU_BATCH_SIZE \
    --per_gpu_eval_batch_size=$GPU_BATCH_SIZE \
    --save_steps=100000 \
    --overwrite_output_dir \
> ../log/lm/no_fair_mask/res_{$MODEL_TYPE}_256_Batch_{$GPU_BATCH_SIZE}_Sample_{$SUB_SAMPLE}_epoch_{$EPOCHS}.txt &
