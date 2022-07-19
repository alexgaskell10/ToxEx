export CUDA_VISIBLE_DEVICES=2
task=hf_clm_with_demos
model_dir=/data2/ag/home/ag/experiments/gpt3-explanation-student

for ext in hf_clm_with_demos_gpt2-large hf_clm_with_demos_gpt2 hf_clm_with_demos_gpt2-medium
do
    echo 'model: '$ext
    model=$model_dir/$ext
    # For obtaining preditions set bsz = 1
    python language-modeling/run_clm.py \
        --model_name_or_path $model \
        --validation_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/seed.csv \
        --max_eval_samples 20 \
        --output_dir $model/eval_predictions \
        --per_device_eval_batch_size 1 \
        --do_eval \
        --label_smoothing_factor 0.1 \
        --logging_steps 1 \
        --overwrite_output_dir \
        --num_beams 5 \
        --generation_max_length 512 \
        --fp16 \
        --set_special_tokens \
        --predict_with_generate

    # Obtain predictions
    python language-modeling/run_clm.py \
        --model_name_or_path $model \
        --validation_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/seed.csv \
        --predict_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/seed.csv \
        --output_dir $model/seed_predictions \
        --per_device_eval_batch_size 1 \
        --do_eval \
        --do_predict \
        --max_eval_samples 1 \
        --label_smoothing_factor 0.1 \
        --logging_steps 1 \
        --overwrite_output_dir \
        --num_beams 5 \
        --generation_max_length 768 \
        --fp16 \
        --set_special_tokens \
        --predict_with_generate
done



# export CUDA_VISIBLE_DEVICES=2
# task=hf_clm_wo_demos
# model_dir=/data2/ag/home/ag/experiments/gpt3-explanation-student

# for ext in hf_clm_wo_demos_gpt2-large
# do
#     echo 'model: '$ext
#     model=$model_dir/$ext
#     python language-modeling/run_clm.py \
#         --model_name_or_path $model \
#         --validation_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/dev.csv \
#         --predict_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/$task/dev.csv \
#         --output_dir $model/full_predictions \
#         --per_device_eval_batch_size 1 \
#         --do_eval \
#         --do_predict \
#         --max_eval_samples 1 \
#         --label_smoothing_factor 0.1 \
#         --logging_steps 1 \
#         --overwrite_output_dir \
#         --num_beams 5 \
#         --generation_max_length 512 \
#         --fp16 \
#         --set_special_tokens \
#         --predict_with_generate
# done