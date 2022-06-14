import os, sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind

from common import (
    add_prefix,
    load_data,
    label_map,
    answer_cols,
    retain_worker_ids,
    fleiss_kappa,
)
from krippendorffs_alpha import krippendorff_alpha, interval_metric, nominal_metric


df_samples, df_output, df_hits, df_hits_unag, df_answers, df_answers_unag = load_data(
    filter_workers=True)


def annotator_headline_stats(df_answers):
    # Aggregated annotator performance
    df_annot_perf = df_answers[['avg_response', 'majority_numeric']].dropna()
    df_annot_perf['avg_response_bin'] = df_annot_perf['avg_response'].apply(
            lambda x: 1 if x > 0.67 else 0 if x > 0.33 else -1)
    df_annot_perf['avg_annotator_bin'] = df_annot_perf['majority_numeric'].apply(
            lambda x: 1 if x > 0.67 else 0 if x > 0.33 else -1)
    correct = df_annot_perf['avg_annotator_bin'] == df_annot_perf['avg_response_bin']
    print('Aggregated headline statistice:')
    print('Accuracy of majority vote reponses against gold responses (grouping into [\"positive\", \"neutral\", \"bad\"]:')
    print('Mean:', correct.mean())
    print('Std:', correct.std())


def performance_by_dataset(df_answers):
    # Performance by prompt for aggregated annotator responses
    df_prompt_perf = df_answers[['dataset', 'median_annotator', 'prompt_type']].dropna()
    df_prompt_perf['annotator_mean'] = df_answers['median_annotator']
    df_prompt_perf['annotator_std'] = df_answers['median_annotator']
    res = df_prompt_perf.groupby('dataset').aggregate(
        {'annotator_mean':np.mean, 'annotator_std':np.std, 'median_annotator':len})
    print("Performance by prompt:")
    print(res)


def performance_by_prompt(df_answers):
    # Performance by prompt for aggregated annotator responses
    df_prompt_perf = df_answers[['prompt_type', 'median_annotator']].dropna()
    df_prompt_perf['annotator_mean'] = df_answers['median_annotator']
    df_prompt_perf['annotator_std'] = df_answers['median_annotator']
    res = df_prompt_perf.groupby('prompt_type').aggregate(
        {'annotator_mean':np.mean, 'annotator_std':np.std, 'median_annotator':len})
    print("Performance by prompt:")
    print(res)


def compute_krippendorffs_alpha(df_answers_unag, metric=nominal_metric, do_agg=False, use_norm=False):
    # Put data into correct format:
    # [
    #     {unit1:value, unit2:value, ...},  # coder 1
    #     {unit1:value, unit3:value, ...},   # coder 2
    #     ...                            # more coders
    # ]
    key = 'znorm_numeric_annotator_0' if use_norm else 'numeric_annotator_0'

    data = df_answers_unag[['WorkerId', 'prompt_type', 'uid', key, ]]
    data = data[~data[key].isnull()].drop('prompt_type', axis=1)

    # Map to indices
    if do_agg:
        data['annotator_class'] = data[key].apply(lambda x: 
            # 1 if x > 0.5 else 0)
            # 1 if x > 0.67 else 0 if x > 0.33 else -1)
            1 if x > 0.67 else 0.5 if x > 0.33 else 0)
    else:
        data['annotator_class'] = data[key] #(data[key] * 4 + 1).astype(int)
    
    rows = data.pivot(index='WorkerId', columns='uid', values='annotator_class')
    rows = rows.fillna('*').astype(str).to_dict(orient='records')
    
    res = krippendorff_alpha(rows, metric=metric, missing_items='*')

    print(f'Krippendorffs alpha for {"grouped" if do_agg else "ungrouped"} data:', res)
    print('Done')


def compute_krippendorffs_alpha_by_prompt(df_answers_unag, do_agg=False):
    for i in range(1,4):
        print('Prompt type:', i)
        compute_krippendorffs_alpha(
            df_answers_unag[df_answers_unag['prompt_type']==i], 
            do_agg=do_agg)


def compute_krippendorffs_alpha_by_dataset(df_answers_unag, do_agg=False):
    for dset in ['sbf','unib','mhs']:
        print('Dataset:', dset)
        compute_krippendorffs_alpha(
            df_answers_unag[df_answers_unag['dataset_x']==dset], 
            do_agg=do_agg)


def sample_level_performance(df_answers):

    df_ids = df_answers[['id', 'text', 'prompt_type', 'majority_numeric']
        ].dropna().pivot(index=['id', 'text'], columns='prompt_type').dropna().reset_index()
    df_ids.columns = df_ids.columns.droplevel(0)
    df_ids['2-1'] = df_ids[2.0] - df_ids[1.0]
    df_ids['3-1'] = df_ids[3.0] - df_ids[1.0]
    df_ids['3-2'] = df_ids[3.0] - df_ids[2.0]
    df_ids['1-2_max'] = df_ids[[1.0,2.0]].max(axis=1)
    df_ids['1-3_max'] = df_ids[[1.0,2.0]].max(axis=1)
    df_ids['1-2-3_max'] = df_ids[[1.0,2.0,3.0]].max(axis=1)
    df_ids['2-1_bin'] = df_ids['2-1'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    df_ids['3-1_bin'] = df_ids['3-1'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    df_ids['3-2_bin'] = df_ids['3-2'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    delta_ids = df_ids.sort_values('2-1').iloc[:10,0].tolist()
    df_answers[df_answers['id'].isin(delta_ids[0:1])][
        ['text','prompt_type','target_group_str','explanation','majority_numeric']
        ].sort_values('prompt_type').values

    def ttest(col_1, col_2):
        return ttest_ind(
            df_ids[col_1].dropna(),
            df_ids[col_2].dropna(),
            equal_var=False
        )

    print(1, 3, ttest(1.0, 2.0))
    print(1, 3, ttest(1.0, 3.0))
    print('1-2_max', ttest(1.0, '1-2_max'))
    print('1-3_max', ttest(1.0, '1-3_max'))
    print('1-2-3_max', ttest(1.0, '1-2-3_max'))


def ensemble_performance(df_answers):

    df_ids = df_answers[['id', 'text', 'prompt_type', 'median_annotator']].dropna(
        ).pivot(index=['id', 'text'], columns='prompt_type').dropna().reset_index()
    df_ids.columns = df_ids.columns.droplevel(0)
    df_ids['2-1'] = df_ids[2.0] - df_ids[1.0]
    df_ids['3-1'] = df_ids[3.0] - df_ids[1.0]
    df_ids['3-2'] = df_ids[3.0] - df_ids[2.0]
    df_ids['1-2_max'] = df_ids[[1.0,2.0]].max(axis=1)
    df_ids['1-3_max'] = df_ids[[1.0,2.0]].max(axis=1)
    df_ids['1-2-3_max'] = df_ids[[1.0,2.0,3.0]].max(axis=1)
    df_ids['2-1_bin'] = df_ids['2-1'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    df_ids['3-1_bin'] = df_ids['3-1'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    df_ids['3-2_bin'] = df_ids['3-2'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    delta_ids = df_ids.sort_values('2-1').iloc[:10,0].tolist()
    df_answers[df_answers['id'].isin(delta_ids[0:1])][
        ['text','prompt_type','target_group_str','explanation','mean_numeric_annotator']
        ].sort_values('prompt_type').values
    print(df_ids[[1.0,2.0,3.0,'1-2_max','1-3_max','1-2-3_max']].mean())
    print((df_ids[[1.0,2.0,3.0,'1-2_max','1-3_max','1-2-3_max']] > 0.5).mean())
    print('Done')


def qualitative_agreement_analysis(df_answers):
    data = df_answers[['HITId','numeric_annotator_0', 'numeric_annotator_1', 'numeric_annotator_2']].dropna()
    data['bin_annotator_0'] = data['numeric_annotator_0'].apply(
        lambda x: 1 if x > 0.67 else 0 if x > 0.33 else -1)
    data['bin_annotator_1'] = data['numeric_annotator_1'].apply(
        lambda x: 1 if x > 0.67 else 0 if x > 0.33 else -1)
    data['bin_annotator_2'] = data['numeric_annotator_2'].apply(
        lambda x: 1 if x > 0.67 else 0 if x > 0.33 else -1)
    data['ungrouped_disagree'] = data[['numeric_annotator_0', 'numeric_annotator_1', 'numeric_annotator_2']].apply(
        lambda rows: len(rows.unique()) == len(rows), axis=1) 
    data['grouped_disagree'] = data[['bin_annotator_0', 'bin_annotator_1', 'bin_annotator_2']].apply(
        lambda rows: len(rows.unique()) == len(rows), axis=1) 
    ids = data[data['grouped_disagree']]['HITId']

    filtered = df_answers[['HITId',
        'numeric_annotator_0', 'numeric_annotator_1', 'numeric_annotator_2', 
        'prompt', 'prompt_type', 'explanation']
    ]
    rows = filtered[filtered['HITId'].isin(ids)].drop('HITId', axis=1)
    # rows.to_csv('data/mturk/disagreement.csv')
    print('Done')


if __name__ == '__main__':
    # compute_krippendorffs_alpha_by_dataset(df_answers_unag)
    # compute_krippendorffs_alpha_by_prompt(df_answers_unag)
    # qualitative_agreement_analysis(df_answers)
    # performance_by_dataset(df_answers)
    # annotator_headline_stats(df_answers)
    # performance_by_prompt(df_answers)
    compute_krippendorffs_alpha(df_answers_unag, metric=interval_metric, do_agg=True)
    compute_krippendorffs_alpha(df_answers_unag, metric=interval_metric, do_agg=False)
    # sample_level_performance(df_answers)
    # ensemble_performance(df_answers)
    print('Done')

    # TODO:
