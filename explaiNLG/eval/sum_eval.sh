export CUDA_VISIBLE_DEVICES=2

# for model in hf_clm_wo_demos_gpt2-large
# do
#     echo $model
#     python summarization/eval_gens.py \
#         --generations_file /data2/ag/home/ag/experiments/gpt3-explanation-student/$model/full_predictions/generated_predictions.txt \
#         --references_file /data2/ag/home/ag/datasets/data-aux/gpt3_explanations/all_gens_only/dev.csv \
#         --batch_size 1 \
#         --write_all
# done

python /data2/ag/home/explaiNLG/eval/eval_gens.py \
    --generations_file /data2/ag/home/ToxEx/data/auto_eval/cands.txt \
    --references_file /data2/ag/home/ToxEx/data/auto_eval/df.csv \
    --batch_size 1 \
    --write_all \
    --summary_column explanation_y
