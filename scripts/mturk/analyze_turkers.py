import os, sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from common import (
    add_prefix,
    load_data,
    label_map,
    answer_cols,
    retain_worker_ids,
    fleiss_kappa
)

fig_dir = Path('scripts/mturk/figs')
fig_dir.mkdir(exist_ok=True)

df_samples, df_output, df_hits, df_hits_unag, df_answers, df_answers_unag = load_data(
    filter_workers=True)


def annotations_hist(df_answers_unag):
    # How many samples does each turker do?
    df_annot_counts = df_answers_unag[['WorkerId','annotator_0']].dropna(
        ).groupby('WorkerId').count()

    plt.figure(figsize=(8, 4))
    plt.hist(df_annot_counts['annotator_0'], bins=20)
    plt.xlabel("Completed Samples")
    plt.ylabel("Number of Turkers")
    plt.yticks(list(range(0, 32, 4)))
    plt.savefig(fig_dir / 'turker_hist_all.png')
    plt.close()

    print("Average number of annotations:", df_annot_counts.mean())


def gold_turker_analysis(df_answers_unag):
    # How many gold annotations does each turker do?
    df_annot_perf = df_answers_unag[['WorkerId','avg_response','numeric_annotator_0']].dropna()
    df_annot_perf['avg_response_bin'] = df_annot_perf['avg_response'].apply(
        lambda x: 1 if x > 0.67 else 0 if x > 0.33 else -1)
    df_annot_perf['avg_annotator_bin'] = df_annot_perf['numeric_annotator_0'].apply(
        lambda x: 1 if x > 0.67 else 0 if x > 0.33 else -1)
    df_annot_perf['correct'] = df_annot_perf['avg_annotator_bin'] == df_annot_perf['avg_response_bin']
    # df_annot_perf[['WorkerId', 'numeric_annotator_0']].groupby('WorkerId').aggregate({'numeric_annotator_0':np.mean, 'numeric_annotator_0':len})

    # Identify good workers
    worker_perf = df_annot_perf[['WorkerId', 'correct']].groupby('WorkerId'
        ).aggregate([np.mean, len]).sort_values(('correct','len'), ascending=False)
    print('Worker performance vs gold annotations')
    print(worker_perf)
    retain_worker_ids = worker_perf[worker_perf.iloc[:,0] >= 0.6].index.tolist()
    worker_ranking = worker_perf[(worker_perf[('correct','len')] > 5)
        ].sort_values(('correct','mean'), ascending=False).index.tolist()
    # worker_perf[~worker_perf.index.isin(['AJKHQUPAKCEE6', 'A4D99Y82KOLC8','A3DTX4Z9Z8FBVC'])]

    plt.figure(figsize=(8, 4))
    plt.hist(worker_perf[('correct', 'len')], bins=20)
    plt.xlabel("Completed Gold Samples")
    plt.ylabel("Number of Turkers")
    plt.yticks(list(range(0, 25, 4)))
    plt.savefig(fig_dir / 'turker_hist_gold.png')
    plt.close()

    print("Average number of gold annotations:", worker_perf[('correct', 'len')].mean())

    plt.figure(figsize=(8, 4))
    plt.hist(worker_perf[('correct', 'mean')], bins=20)
    plt.xlabel("Gold Agreement")
    plt.ylabel("Number of Turkers")
    # plt.yticks(list(range(0, 32, 4)))
    plt.savefig(fig_dir / 'turker_performance.png')
    plt.close()

    print("Average annotation accuracy (random = 33%):", df_annot_perf['correct'].mean())

    # Stacked bar chart of gold performance by annotator
    data = worker_perf.sort_values(('correct','len'), ascending=False)
    data.columns = data.columns.droplevel(0)
    data['num_correct'] = (data['mean'] * data['len']).astype(int)
    data['num_incorrect'] = ((1-data['mean']) * data['len']).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(data.index, data['num_correct'], label='Correct')
    ax.bar(data.index, data['num_incorrect'], label='Incorrect', bottom=data['num_correct'])
    plt.xlabel("Performance by turker")
    plt.ylabel("Turkers")
    # plt.yticks(list(range(0, 32, 4)))
    ax.legend(loc=(0.8, 0.75))
    plt.xticks(fontsize=8, rotation=60)
    plt.savefig(fig_dir / 'turker_performance_breakdown.png')
    plt.close()


def inter_annotator_agreement(df_answers):
    # Compute fleiss kappa on Likert scale responses
    # df_kappa_tmp = df_answers[['prompt_type','dataset']+answer_cols].dropna()
    df_kappa_tmp = df_answers[['prompt_type','dataset']+['annotator_0','annotator_1','annotator_2']]
    freq_counts = [{k:row.tolist().count(k) for k in label_map} for row in df_kappa_tmp.values]
    df_kappa = pd.DataFrame.from_dict(freq_counts)
    fleiss_kappa_res = fleiss_kappa(df_kappa.to_numpy())
    print("Fleiss kappa on raw Likert scale responses:\n", fleiss_kappa_res)
    
    # # Fleiss kappa by prompt type
    # df_kappa['prompt_type'] = df_kappa_tmp['prompt_type'].reset_index(drop=True)
    # fleiss_kappa_res_1 = fleiss_kappa(df_kappa[df_kappa['prompt_type']==1.0].drop('prompt_type', axis=1).to_numpy())
    # fleiss_kappa_res_2 = fleiss_kappa(df_kappa[df_kappa['prompt_type']==2.0].drop('prompt_type', axis=1).to_numpy())
    # fleiss_kappa_res_3 = fleiss_kappa(df_kappa[df_kappa['prompt_type']==3.0].drop('prompt_type', axis=1).to_numpy())
    # df_kappa.drop('prompt_type', axis=1, inplace=True)
    # print("Fleiss kappa on raw Likert scale responses for prompt type 1:\n", fleiss_kappa_res_1)
    # print("Fleiss kappa on raw Likert scale responses for prompt type 2:\n", fleiss_kappa_res_2)
    # print("Fleiss kappa on raw Likert scale responses for prompt type 3:\n", fleiss_kappa_res_3)
    
    # # Fleiss kappa by prompt type
    # df_kappa['dataset'] = df_kappa_tmp['dataset'].reset_index(drop=True)
    # fleiss_kappa_res_1 = fleiss_kappa(df_kappa[df_kappa['dataset']=='sbf'].drop('dataset', axis=1).to_numpy())
    # fleiss_kappa_res_2 = fleiss_kappa(df_kappa[df_kappa['dataset']=='mhs'].drop('dataset', axis=1).to_numpy())
    # fleiss_kappa_res_3 = fleiss_kappa(df_kappa[df_kappa['dataset']=='unib'].drop('dataset', axis=1).to_numpy())
    # df_kappa.drop('dataset', axis=1, inplace=True)
    # print("Fleiss kappa on raw Likert scale responses for sbf:\n", fleiss_kappa_res_1)
    # print("Fleiss kappa on raw Likert scale responses for mhs:\n", fleiss_kappa_res_2)
    # print("Fleiss kappa on raw Likert scale responses for unib:\n", fleiss_kappa_res_3)

    # Compute fleiss kappa on 3-grouped likert responses
    # i.e. map to ["positive", "neutral", "negative"]
    df_kappa['grouped_bad'] = df_kappa[['strongly disagree', 'weakly disagree']].sum(1)
    df_kappa['grouped_good'] = df_kappa[['strongly agree', 'weakly agree']].sum(1)
    df_kappa['prompt_type'] = df_kappa_tmp['prompt_type'].reset_index(drop=True)
    fleiss_kappa_grouped = fleiss_kappa(
        df_kappa[['grouped_good', 'neutral', 'grouped_bad']].to_numpy())
    print("Fleiss kappa on 3-grouped responses:\n", fleiss_kappa_grouped)

    # Compute fleiss kappa on 2-grouped likert responses
    # i.e. map to ["positive", "negative"]
    df_kappa['grouped_bad'] = df_kappa[['strongly disagree', 'weakly disagree', 'neutral']].sum(1)
    df_kappa['grouped_good'] = df_kappa[['strongly agree', 'weakly agree']].sum(1)
    df_kappa['prompt_type'] = df_kappa_tmp['prompt_type'].reset_index(drop=True)
    fleiss_kappa_grouped = fleiss_kappa(
        df_kappa[['grouped_good', 'grouped_bad']].to_numpy())
    print("Fleiss kappa on 2-grouped responses:\n", fleiss_kappa_grouped)

    print('Done')


def cheat_analysis(df_answers_unag):
    data = df_answers_unag[['WorkerId','SubmitTime','AcceptTime']].dropna()
    data['AnnotationTime'] = (data['SubmitTime'] - data['AcceptTime']).apply(
        lambda x: x.seconds)
    grouped = data[['WorkerId','AnnotationTime']].groupby('WorkerId').aggregate(
        {'AnnotationTime':np.mean, 'WorkerId':len})
    grouped.columns = ['MeanSeconds','NumAnnots']
    grouped.sort_values('NumAnnots', ascending=False, inplace=True)
    # print(grouped)

    problem_wids = ['AJKHQUPAKCEE6', 'A4D99Y82KOLC8','A3DTX4Z9Z8FBVC']
    all_rows = data[~data['WorkerId'].isin(problem_wids)]['AnnotationTime'].sort_values()
    all_rows = trim_outliers(all_rows, 5)

    for wid in problem_wids:
        wid_rows = data[data['WorkerId'] == wid]['AnnotationTime'].sort_values()
        wid_rows = trim_outliers(wid_rows, 5)
        print(wid, 'Worker mean time (s):', wid_rows.mean(), 'Overall mean time (s):', all_rows.mean(), ttest_ind(wid_rows, all_rows))
        
    print('Done')


def trim_outliers(sorted_data, percent):
    n = len(sorted_data)
    outliers = n*percent / 100
    trimmed_data = sorted_data[int(outliers): int(n-outliers)]
    return trimmed_data


def turker_positivity(df_answers_unag):
    data = df_answers_unag[['WorkerId','numeric_annotator_0']].dropna()
    data['std'] = data['numeric_annotator_0']
    data['annotations'] = data['numeric_annotator_0']
    res = data.groupby(['WorkerId']).aggregate(
        {'numeric_annotator_0':np.mean, 'std':np.std, 'annotations': len})
    print('Average score per annotator:')
    print(res)
    # res.to_csv('scripts/mturk/turker_mean_scores.csv')
    print('Done')


if __name__ == '__main__':
    # turker_positivity(df_answers_unag)
    # cheat_analysis(df_answers_unag)
    # annotations_hist(df_answers_unag)
    gold_turker_analysis(df_answers_unag)
    inter_annotator_agreement(df_answers)
    print('Done')
