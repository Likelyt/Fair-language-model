export TRAIN_FILE=../data/wikitext-103-raw/wili.train.raw
export TEST_FILE=../data/wikitext-103-raw/wiki.test.raw
export OUTPUT_DIR=../output/pretrained/lm/block-256/fair_mask/mask_1
export MASK=True
export GPU_BATCH_SIZE=12
export MODEL_TYPE=gpt2
export SUB_SAMPLE=1
export MASK_PROB=1
export EPOCHS=2

nohup python lm_finetuning.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$MODEL_TYPE \
    --model_name_or_path=gpt2 \
    --block_size=256 \
    --sub_sample=$SUB_SAMPLE \
    --mask_fair \
    --mask_prob=$MASK_PROB \
    --mlm \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs=$EPOCHS \
    --per_gpu_train_batch_size=$GPU_BATCH_SIZE \
    --per_gpu_eval_batch_size=$GPU_BATCH_SIZE \
    --save_steps=20000 \
    --overwrite_output_dir \
> ../log/lm/fair/res_{$MODEL_TYPE}_256_Batch_{$GPU_BATCH_SIZE}_mask_{$MASK}_prob_{$MASK_PROB}_Sample_{$SUB_SAMPLE}_epoch_{$EPOCHS}.txt &