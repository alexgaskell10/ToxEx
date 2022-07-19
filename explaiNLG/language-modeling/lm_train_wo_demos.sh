export CUDA_VISIBLE_DEVICES=2
task=hf_clm_wo_demos
model=gpt2-medium
cleaned_model_name=$(echo $model | sed 's/\//_/')       # Replace "/" with "_"

python language-modeling/run_clm.py \
    --model_name_or_path $model \
    --train_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/train.csv \
    --validation_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/dev.csv \
    --predict_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/dev.csv \
    --max_eval_samples 100 \
    --output_dir /data2/ag/home/ag/experiments/gpt3-explanation-student/$task'_'$cleaned_model_name \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --label_smoothing_factor 0.1 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --num_beams 5 \
    --generation_max_length 512 \
    --fp16 \
    --set_special_tokens \
    --predict_with_generate

