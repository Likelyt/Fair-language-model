export TRAIN_FILE='openwebtext'
export VAL_FILE='../data/wikitext-103-raw/wiki.test.raw'
export OUTPUT_DIR=../output/
export MODEL_MAIN='gpt2-main'
export MODEL_AUX='gpt2-aux'  # roberta-base-aux/gpt2-aux
export CLS_SELECTION_METHOD='[CLS]_TOKEN'
export SUB_SAMPLE_SIZE=50000
export ALPHA=3
export LOAD_PRETRAINED_CLS='pretrain' # init/pretrain
export PER_GPU_BATCH_SIZE=16 # 8,12,16
export SWTICH_FREQ_DIS=20 # 40,60,80
export SWTICH_FREQ_GEN=20
export EPOCHS=2

nohup python main_adv.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=base \
    --model_name_or_path_main=$MODEL_MAIN \
    --model_name_or_path_aux=$MODEL_AUX \
    --block_size=256 \
    --cls_selection_method=$CLS_SELECTION_METHOD \
    --load_pretrained_cls=$LOAD_PRETRAINED_CLS \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$VAL_FILE \
    --num_train_epochs=$EPOCHS \
    --per_gpu_train_batch_size=$PER_GPU_BATCH_SIZE \
    --per_gpu_eval_batch_size=$PER_GPU_BATCH_SIZE \
    --alpha=$ALPHA \
    --sub_sample_size=$SUB_SAMPLE_SIZE \
    --switch_freq_dis=$SWTICH_FREQ_DIS \
    --switch_freq_gen=$SWTICH_FREQ_GEN \
> ../log/adv/res_09_15_{$LOAD_PRETRAINED_CLS}_A_{$ALPHA}_B_{$PER_GPU_BATCH_SIZE}_CLS_{$MODEL_AUX}_Sample_{$SUB_SAMPLE_SIZE}_remove_0_switch_{$SWTICH_FREQ_DIS}_{$SWTICH_FREQ_GEN}.txt &