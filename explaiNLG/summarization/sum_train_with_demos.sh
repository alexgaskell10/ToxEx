export CUDA_VISIBLE_DEVICES=0
task=hf_encdec_with_demos

for model in t5-large
do
    python explaiNLG/summarization/run_summarization.py \
        --model_name_or_path $model \
        --do_train \
        --do_eval \
        --do_predict \
        --train_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/train.csv \
        --validation_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/dev.csv \
        --test_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/dev.csv \
        --output_dir /data2/ag/home/ag/experiments/gpt3-explanation-student/$task'_'$model'_tmp' \
        --max_eval_samples 10000 \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --gradient_accumulation_steps=32 \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 128 \
        --evaluation_strategy epoch \
        --num_train_epochs 5 \
        --learning_rate 5e-5 \
        --label_smoothing_factor 0.1 \
        --logging_steps 1 \
        --save_total_limit 1 \
        --predict_with_generate

done