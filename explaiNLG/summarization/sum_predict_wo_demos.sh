source explaiNLG/summarization/set_vars.sh

export CUDA_VISIBLE_DEVICES=0
task=hf_encdec_wo_demos
model_dir=/data2/ag/home/ag/experiments/gpt3-explanation-student

for ext in hf_encdec_wo_demos_t5-large #hf_encdec_wo_demos_t5-large
do
    echo 'model: '$ext
    model=$model_dir/$ext
    # # For obtaining inference time with bsz = 1
    # python explaiNLG/summarization/run_summarization.py \
    #     --model_name_or_path $model \
    #     --validation_file $DATA_DIR/$task/seed.csv \
    #     --max_eval_samples 20 \
    #     --output_dir $model/eval_predictions \
    #     --per_device_eval_batch_size 1 \
    #     --do_eval \
    #     --label_smoothing_factor 0.1 \
    #     --logging_steps 1 \
    #     --overwrite_output_dir \
    #     --num_beams 5 \
    #     --max_source_length 1024 \
    #     --max_target_length 128 \
    #     --predict_with_generate

    # Obtain predictions
    python explaiNLG/summarization/run_summarization.py \
        --model_name_or_path $model \
        --validation_file $DATA_DIR/$task/demo-samples.csv \
        --test_file $DATA_DIR/$task/demo-samples.csv \
        --output_dir $model/retry \
        --per_device_eval_batch_size 1 \
        --do_eval \
        --do_predict \
        --max_eval_samples 1 \
        --label_smoothing_factor 0.1 \
        --logging_steps 1 \
        --overwrite_output_dir \
        --num_beams 5 \
        --max_source_length 512 \
        --max_target_length 128 \
        --predict_with_generate
done
        # --device_map '{0: [0, 1, 2, 3, 4, 5, 6], 1: [7, 8, 9, 10, 11, 12, 13, 14, 15], 2: [16, 17, 18, 19], 3: [20, 21, 22, 23]}' \