export CUDA_VISIBLE_DEVICES=0
task=hf_encdec_with_demos
model_dir=/data2/ag/home/ag/experiments/gpt3-explanation-student

for ext in hf_encdec_with_demos_t5-large
do
    echo 'model: '$ext
    model=$model_dir/$ext
    # # For obtaining inference time with bsz = 1
    # python explaiNLG/summarization/run_summarization.py \
    #     --model_name_or_path $model \
    #     --validation_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/seed.csv \
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
        --validation_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/demo-samples.csv \
        --test_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/demo-samples.csv \
        --output_dir $model/retry \
        --per_device_eval_batch_size 1 \
        --do_eval \
        --do_predict \
        --max_eval_samples 1 \
        --label_smoothing_factor 0.1 \
        --logging_steps 1 \
        --overwrite_output_dir \
        --num_beams 5 \
        --max_source_length 1024 \
        --max_target_length 128 \
        --predict_with_generate
done
