export CUDA_VISIBLE_DEVICES=0,1,2,3
task=hf_encdec_with_demos

for model in t5-3b
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
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 32 \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 128 \
        --evaluation_strategy epoch \
        --num_train_epochs 5 \
        --learning_rate 1e-5 \
        --label_smoothing_factor 0.1 \
        --logging_steps 1 \
        --device_map '{0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [8, 9, 10, 11, 12, 13, 14, 15], 2: [16, 17, 18, 19], 3: [20, 21, 22, 23]}' \
        --save_total_limit 5 \
        --gradient_checkpointing \
        --predict_with_generate


done
