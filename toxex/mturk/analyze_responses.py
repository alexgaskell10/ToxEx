import os, sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind

from .common import (
    add_prefix,
    load_data,
    label_map,
    answer_cols,
    retain_worker_ids,
    fleiss_kappa,
    group,
    worker_ranking,
)
from .krippendorffs_alpha import krippendorff_alpha, interval_metric, nominal_metric


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


def compute_krippendorffs_alpha(df_answers_unag, metric=nominal_metric, agg_level=2, use_norm=False):
    # Put data into correct format:
    # [
    #     {unit1:value, unit2:value, ...},  # coder 1
    #     {unit1:value, unit3:value, ...},   # coder 2
    #     ...                            # more coders
    # ]
    key = 'znorm_numeric_annotator_0' if use_norm else 'numeric_annotator_0'

    data = df_answers_unag[['WorkerId', 'prompt_type', 'uid', key]]
    data = data[~data[key].isnull()].drop('prompt_type', axis=1)

    # Map to indices
    data['annotator_class'] = data[key].apply(lambda x: group(x, agg_level))
    
    rows = data.drop_duplicates(['WorkerId','uid']).pivot(index='WorkerId', columns='uid', values='annotator_class')
    rows = rows.fillna('*').astype(str).to_dict(orient='records')
    
    res = krippendorff_alpha(rows, metric=metric, missing_items='*')

    print(f'Krippendorffs alpha for {agg_level} likert scale data:', res)
    
    return {f'{metric.__name__}_{agg_level}-level': res}


def compute_krip_alpha(df_answers_unag):
    results = {}
    for metric in [nominal_metric, interval_metric]:
        for agg_level in [2,3,5]:
            results = {**results, **compute_krippendorffs_alpha(df_answers_unag, metric, agg_level)}
    return {'ka_'+k:v for k,v in results.items()}


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


def sample_level_performance(df_answers, agg_level=2):

    df_answers['median_annotator'] = df_answers['median_annotator'].apply(
        lambda x: group(x, agg_level))
    df_ids = df_answers.reset_index()[['id','text','prompt_type','median_annotator','dataset']
        ].dropna().pivot(
            index=['id', 'text', 'dataset'], columns='prompt_type').dropna()
    df_ids.columns = df_ids.columns.droplevel(0)
    df_ids = df_ids[[ix[2] != 'lfw' for ix in df_ids.index]].reset_index()

    df_ids['1-2_max'] = df_ids[[1.0,2.0]].max(axis=1)
    df_ids['1-3_max'] = df_ids[[1.0,3.0]].max(axis=1)
    df_ids['1-2-3_max'] = df_ids[[1.0,2.0,3.0]].max(axis=1)
    # df_ids['2-1_bin'] = df_ids['2-1'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    # df_ids['3-1_bin'] = df_ids['3-1'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    # df_ids['3-2_bin'] = df_ids['3-2'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)

    # df_ids['2-1'] = df_ids[2.0] - df_ids[1.0]
    # df_ids['3-1'] = df_ids[3.0] - df_ids[1.0]
    # df_ids['3-2'] = df_ids[3.0] - df_ids[2.0]
    # delta_ids = df_ids.sort_values(['2-1',3.0]).iloc[:20,0].tolist()
    # def res(i):
    #     return df_answers[df_answers['id'].isin(delta_ids[i:i+1])][
    #         ['text','prompt_type','target_group_str','explanation','median_annotator']
    #         ].sort_values('prompt_type').values
    # Examples: 'sbf-44667'

    def ttest(df, col_1, col_2):
        return ttest_ind(
            df[col_1],
            df[col_2],
            equal_var=False
        )

    res = df_ids.groupby('dataset').mean()
    all = pd.DataFrame(df_ids.mean()).transpose()
    all.index = ['all']
    res = pd.concat((res, all)).transpose()

    ttest(df_ids, 1.0, 2.0)

    rows = []
    for row in res.index:
        row_vals = []
        for col in res.columns:
            if col == 'all':
                continue
            row_vals.append(ttest(df_ids[df_ids['dataset']==col], row, 1.0).pvalue)
        rows.append(row_vals)

    return res
        

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


def missing_annotations(df_answers_unag):
    data = df_answers_unag
    data['approved'] = data['WorkerId'].isin(worker_ranking)
    # data['id'] = data['comment_id'] + '::' + data['prompt_type']
    grouped = data[data['approved']][['comment_id','prompt_type','WorkerId']].groupby(['comment_id','prompt_type']).count().dropna()
    grouped['missing'] = 5 - grouped

    # TODO: filter out bad workers and leave remaining rows
    pids = [1.0,2.0,3.0]
    all_rows = list(zip(data['comment_id'],data['prompt_type']))
    missing_rows = data[[r not in grouped.index for r in all_rows]][['comment_id','prompt_type']].dropna().drop_duplicates()
    df_missing = pd.DataFrame({
        'WorkerId':[0 for _ in range(len(missing_rows))],
        'missing':[5 for _ in range(len(missing_rows))],
        'comment_id': [x for x in missing_rows['comment_id']],
        'prompt_type': [x for x in missing_rows['prompt_type']],
    }).dropna()
    df_missing.set_index(['comment_id', 'prompt_type'], inplace=True)
    # grouped = pd.concat((grouped.dropna(), df_missing.dropna()))
    
    need_2 = grouped[grouped['missing'].isin([2,3])].reset_index()
    need_3 = grouped[grouped['missing'].isin([4,5])].reset_index()
    # need_2.reset_index().to_csv()

    all_path = '/data2/ag/home/mturk/round_1/samples/2022-06-07-10-36-32-proc-withannots-3.csv'
    df_need_2 = pd.read_csv(all_path)
    tmp = df_need_2[[r.tolist() in need_2[['comment_id','prompt_type']].values.tolist() for r in df_need_2[['id','prompt_type']].values]]
    df_need_2 = pd.concat((tmp, df_need_2[df_need_2['prompt_type'].isnull()]))
    df_need_2.to_csv('/data2/ag/home/mturk/round_2/samples/2022-06-20_01-36-proc-withannots.csv', index=False)

    all_path = '/data2/ag/home/mturk/round_1/samples/2022-06-07-10-36-32-proc-withannots-3.csv'
    df_need_3 = pd.read_csv(all_path)
    tmp = df_need_3[[r.tolist() in need_3[['comment_id','prompt_type']].values.tolist() for r in df_need_3[['id','prompt_type']].values]]
    df_need_3 = pd.concat((tmp, df_need_3[df_need_3['prompt_type'].isnull()]))
    df_need_3.to_csv('/data2/ag/home/mturk/round_3/samples/2022-06-20_01-36-proc-withannots.csv', index=False)

    print(len(grouped[grouped['missing']>=3]), len(grouped[grouped['missing']<3]))
    print('Done')    


if __name__ == '__main__':

    df_samples, df_output, df_hits, df_hits_unag, df_answers, df_answers_unag = load_data(
        filter_workers=False)

    # compute_krippendorffs_alpha_by_dataset(df_answers_unag)
    # compute_krippendorffs_alpha_by_prompt(df_answers_unag)
    # qualitative_agreement_analysis(df_answers)
    # performance_by_dataset(df_answers)
    # annotator_headline_stats(df_answers)
    # performance_by_prompt(df_answers)
    # compute_krippendorffs_alpha(df_answers_unag, metric=interval_metric, do_agg=True)
    # compute_krippendorffs_alpha(df_answers_unag, metric=interval_metric, do_agg=False)
    # sample_level_performance(df_answers)
    # ensemble_performance(df_answers)
    missing_annotations(df_hits_unag)

    print('Done')

    # TODO:
