import os, sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from .common import (
    add_prefix,
    load_data,
    label_map,
    answer_cols,
    retain_worker_ids,
    fleiss_kappa,
    group,
    weighted_fleiss_kappa,
)

fig_dir = Path('toxex/mturk/figs')
fig_dir.mkdir(exist_ok=True)


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
    data = worker_perf.sort_values(('correct','len'), ascending=False).reset_index(drop=True)
    data.columns = data.columns.droplevel(0)
    data['num_correct'] = (data['mean'] * data['len']).astype(int)
    data['num_incorrect'] = ((1-data['mean']) * data['len']).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    # colours = ['blue' if i else 'red' for i in data.index.isin(retain_worker_ids)]
    ax.bar(data.index, data['num_correct'], label='Correct')
    ax.bar(data.index, data['num_incorrect'], label='Incorrect', bottom=data['num_correct'])
    plt.xlabel("Performance by turker")
    plt.ylabel("Turkers")
    # plt.yticks(list(range(0, 32, 4)))
    ax.legend(loc=(0.8, 0.75))
    # plt.xticks(fontsize=8, rotation=90)
    plt.xticks(range(len(data)))
    plt.savefig(fig_dir / 'turker_performance_breakdown.png')
    plt.close()


def compute_fleiss_kappa(df_answers):
    results = {}
    # Compute fleiss kappa on Likert scale responses
    # df_kappa_tmp = df_answers[['prompt_type','dataset']+[c for c in answer_cols if c in df_answers.columns]].dropna()
    df_answers = df_answers.reset_index()
    # df_kappa_tmp = df_answers[[i for i in df_answers.columns if isinstance(i, int)]]
    cols = [0,1,2]
    df_kappa_tmp = df_answers[['prompt_type']+cols].dropna()
    freq_counts = [{k:row.tolist().count(k) for k in label_map.values()} for row in df_kappa_tmp[cols].values]
    df_kappa = pd.DataFrame.from_dict(freq_counts)
    data = df_kappa.to_numpy()
    fleiss_kappa_res = fleiss_kappa(data)
    print("Fleiss kappa on raw Likert scale responses:\n", fleiss_kappa_res)
    results['5-level'] = fleiss_kappa_res
    weighted_fleiss_kappa_grouped = weighted_fleiss_kappa(df_kappa_tmp[cols])
    print("Weighted Fleiss kappa on 5-grouped responses:\n", weighted_fleiss_kappa_grouped)
    results['5-level_weighted'] = weighted_fleiss_kappa_grouped
    
    # Compute fleiss kappa on 3-grouped likert responses
    # i.e. map to ["positive", "neutral", "negative"]
    df_kappa['grouped_bad'] = df_kappa[[0.00, 0.25]].sum(1)
    df_kappa['grouped_good'] = df_kappa[[0.75, 1.00]].sum(1)
    df_kappa['neutral'] = df_kappa[[0.50]]
    df_kappa['prompt_type'] = df_kappa_tmp['prompt_type'].reset_index(drop=True)
    data = df_kappa[['grouped_good', 'neutral', 'grouped_bad']].to_numpy()
    fleiss_kappa_grouped = fleiss_kappa(data)
    print("Fleiss kappa on 3-grouped responses:\n", fleiss_kappa_grouped)
    results['3-level'] = fleiss_kappa_grouped
    # weighted_fleiss_kappa_grouped = weighted_fleiss_kappa(data)
    # print("Weighted Fleiss kappa on 3-grouped responses:\n", weighted_fleiss_kappa_grouped)
    # results['3-level_weighted'] = weighted_fleiss_kappa_grouped

    # Compute fleiss kappa on 2-grouped likert responses
    # i.e. map to ["positive", "negative"]
    df_kappa['grouped_bad'] = df_kappa[[0.00,0.25,0.50]].sum(1)
    df_kappa['grouped_good'] = df_kappa[[0.75,1.00]].sum(1)
    df_kappa['prompt_type'] = df_kappa_tmp['prompt_type'].reset_index(drop=True)
    data = df_kappa[['grouped_good', 'grouped_bad']].to_numpy()
    fleiss_kappa_grouped = fleiss_kappa(data)
    print("Fleiss kappa on 2-grouped responses:\n", fleiss_kappa_grouped)
    results['2-level'] = fleiss_kappa_grouped
    # weighted_fleiss_kappa_grouped = weighted_fleiss_kappa(data)
    # print("Weighted Fleiss kappa on 2-grouped responses:\n", weighted_fleiss_kappa_grouped)
    # results['2-level_weighted'] = weighted_fleiss_kappa_grouped

    return {'fk_'+k:v for k,v in results.items()}


def cheat_analysis(df_answers_unag):
    data = df_answers_unag[['WorkerId','SubmitTime','AcceptTime']].dropna()
    data['AnnotationTime'] = (data['SubmitTime'] - data['AcceptTime']).apply(
        lambda x: x.seconds)
    grouped = data[['WorkerId','AnnotationTime']].groupby('WorkerId').aggregate(
        {'AnnotationTime':np.mean, 'WorkerId':len})
    grouped.columns = ['MeanSeconds','NumAnnots']
    grouped.sort_values('NumAnnots', ascending=False, inplace=True)
    # print(grouped)

    problem_wids = [    
       'A3DTX4Z9Z8FBVC',
       'AJKHQUPAKCEE6', 
       'A4D99Y82KOLC8',
    ]
    all_rows = data[~data['WorkerId'].isin(problem_wids)]['AnnotationTime'].sort_values()
    all_rows = trim_outliers(all_rows, 5)

    for wid in problem_wids:
        wid_rows = data[data['WorkerId'] == wid]['AnnotationTime'].sort_values()
        wid_rows = trim_outliers(wid_rows, 5)
        print(wid, 'Worker HITs:', wid_rows.count(), 'Worker mean time (s):', wid_rows.mean(), 'Overall mean time (s):', all_rows.mean(), ttest_ind(wid_rows, all_rows))
        
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


def turker_agreement(df_answers, agg_level):
    num_cols = [0,1,2]
    data = df_answers[num_cols]
    for col in num_cols:
        data[col] = data[col].apply(lambda x: group(x, agg_level))

    data['3_equal'] = data.apply(lambda lst: len(lst.unique()) == 1, axis=1)
    data['2_equal'] = data.apply(lambda lst: len(lst.unique()) == len(lst) - 1, axis=1)
    data['0_equal'] = data.apply(lambda lst: len(lst.unique()) == len(lst), axis=1)
    
    return {
        f'agree-3_{agg_level}-level': data['3_equal'].mean(),
        f'agree-2_{agg_level}-level': data['2_equal'].mean(),
        f'agree-0_{agg_level}-level': data['0_equal'].mean(),
    }


def compute_turker_agreement(df_answers):
    results = {}
    for agg_level in [2,3,5]:
        results.update(turker_agreement(df_answers, agg_level))
    return results


def turker_gold_agreement(df_answers_unag, agg_level=3):
    data = df_answers_unag[['WorkerId','avg_response','numeric_annotator_0']].dropna()
    data['accepted_worker'] = data['WorkerId'].isin(retain_worker_ids)
    data['gold'] = data['avg_response'].apply(lambda x: group(x, agg_level))
    data['pred'] = data['numeric_annotator_0'].apply(lambda x: group(x, agg_level))
    data['correct'] = data['pred'] == data['gold']
    return {
        f'main_all-{agg_level}': data['correct'].mean(),
        f'main_passed-{agg_level}': data['correct'][data['accepted_worker']].mean(),
        f'main_failed-{agg_level}': data['correct'][~data['accepted_worker']].mean(),
    }


def compute_turker_gold_agreement(df_answers_unag):
    # Main round
    results = {}
    for agg_level in [2,3]:
        results.update(turker_gold_agreement(df_answers_unag, agg_level))
    
    return results 

    
def find_bad_annots(df_answers_unag, df_answers):
    problem_wids = [    
        'A3DTX4Z9Z8FBVC',       # jerminy11@live.com
        'AJKHQUPAKCEE6', 
        'A4D99Y82KOLC8',
    ]
    problem_wid = problem_wids[2]
    rows = df_answers[df_answers['WorkerIds'].apply(lambda x: problem_wid in x)]
    rows['annot_num'] = df_answers['WorkerIds'].apply(lambda x: x.index(problem_wid) if problem_wid in x else None)
    rows[['text','text_response']].iloc[2]
    
    # examples = {
    #     # All other annotators agreed this was good but you strongly disagreed.
    #     'A3DTX4Z9Z8FBVC': ['unib-6169899::3.0', ['So, if this is accurate, which i doubt, extremists muslims attack and kill britons in britain and there is a backlash against extremist muslims by britons in britain. go figure.',
    #         'The text is Islamophobic to Muslims because it implies that all Muslims are extremists and that they are responsible for terrorist attacks in Britain.']],
    #     # All other annotators agreed this was good but you disagreed.
    #     'AJKHQUPAKCEE6': ['sbf-44473::1.0', ['Remember when muslims massacred americans and george w. bush responded by bringing in more muslims',
    #    'The text is offensive because it stereotypically associates Muslims with terrorism.']],
    #     # All other annotators agreed this was good but you strongly disagreed.
    #     'A4D99Y82KOLC8': ['sbf-42624::3.0', [
    #         'text': 'Roses are red, my dad is black i wonder when he will ever come back',
    #         'explanation': 'The text is racist because it implies that black people are absent fathers.'
    #     ]],
    # }

    print('Done')


if __name__ == '__main__':

    df_samples, df_output, df_hits, df_hits_unag, df_answers, df_answers_unag = load_data(
        filter_workers=True)

    # turker_positivity(df_answers_unag)
    # cheat_analysis(df_answers_unag)
    # annotations_hist(df_answers_unag)
    gold_turker_analysis(df_answers_unag)
    compute_fleiss_kappa(df_answers)
    print('Done')
