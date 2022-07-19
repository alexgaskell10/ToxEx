export CUDA_VISIBLE_DEVICES=0
task=hf_clm_with_demos
model=gpt2
cleaned_model_name=$(echo $model | sed 's/\//_/')       # Replace "/" with "_"

python language-modeling/run_clm.py \
    --model_name_or_path $model \
    --train_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/train.csv \
    --validation_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/dev.csv \
    --predict_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/dev.csv \
    --max_eval_samples 100 \
    --output_dir /data2/ag/home/ag/experiments/gpt3-explanation-student/$task'_'$cleaned_model_name'_tmp' \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --label_smoothing_factor 0.1 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --save_total_limit 1 \
    --num_beams 5 \
    --generation_max_length 512 \
    --set_special_tokens \
    --predict_with_generate

    # --fp16 \
    # Device map for 6B with demos
    # --device_map '{0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 2: [20, 21, 22, 23], 3: [24, 25, 26, 27]}' \
    # --device_map '{0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 1: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}' \