export WHOLE_TRAINING_SIZE='50000'
export LOAD_PRETRAINED_CLS='intialized' # intialized/pretrained
export MODEL_AUX='gpt2-base'
export LOAD_EPOCH=2
export ALPHA=1
export GPU_ID='7'

nohup python FairEvaluation.py \
    --training_sample_size=$WHOLE_TRAINING_SIZE \
    --alpha=$ALPHA \
    --gpu_id=$GPU_ID \
    --load_epoch=$LOAD_EPOCH \
    --load_pretrained_cls=$LOAD_PRETRAINED_CLS \
    --model_name_or_path_aux=$MODEL_AUX \
    --prompt_gender_cut_length=50 \
    --prompt_age_cut_length=100 \
    --prompt_race_cut_length=100 \
    --output_gender_max_length=200 \
    --output_age_max_length=200 \
    --output_race_max_length=200 \
    --samples_race_num=1000 \
    --code_prompt_gender_cut_length=100 \
    --code_output_gender_max_length=200 \
    --code_prompt_race_cut_length=100 \
    --code_output_race_max_length=200 \
> ../log/eval/lang_eval/res_09_07_{$LOAD_PRETRAINED_CLS}_cls_{$MODEL_AUX}_A_{$ALPHA}_Sample_{$WHOLE_TRAINING_SIZE}_remove_zero_load_epo_{$LOAD_EPOCH}.txt &


