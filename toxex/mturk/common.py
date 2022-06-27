import os, sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from itertools import combinations


label_map = {'strongly disagree': 0, 'weakly disagree': 0.25, 'neutral': 0.5, 'weakly agree': 0.75, 'strongly agree': 1}
answer_cols = ['annotator_0', 'annotator_1', 'annotator_2', 'annotator_3', 'annotator_4']
retain_worker_ids = ['A11BS3GFM9T4FJ', 'A13EXIOBU0UVX4', 'A13S25IYJL6AZ9',
       'A13XNZBCRSEGGY', 'A15ERD4HOFEPHM', 'A1CN4UXQNL22OO',
       'A1KOG5Q6XX0DPU', 'A1NF6PELRKACS9', 'A1OPJ5I9BF44QH', 
       'A1YIGXBU6FQQ6C', 'A1YWZXJAPULFVR', 'A29RH02TXLXLSI',
       'A29UN4GHR6J0WS', 'A2HM35CWB7IIFM', 'A2NJK5F728GAQX',
        'A3AT4BYGP2MWOH', 'A3DU10US4YRB8E',
       'A3LCGX74R6XR5I', 'A3MV3PT4TOO69P', 'A3RR91G2KJO9NI',
       'A45G52Z5J7195', 'A7R4F8BRS8ET0', 'AC95JAUAM2L2Z',
       'AJKA9BKC011F2', 'AKF6C62Q8UW2B', 'AL0OO84JETD31', 'ARFDXQ5RCR74G',

    # 'A1AMGHYG5PT0L2', 
    # 'AEN3K2PLE658P',
    # 'A3QO8R7Z93A87K', 
    # 'A2SDOD67560IN8',   
    #    'A1037WWQHXLFA0', 
    #    'A1PBRKFHSF1OF8',
    #    'A2H1RXUC8XNNIW',
    #    'A2U4H12NH6Z4QL', 
    #    'A26Z3YPJ8RYJQJ', 'AVEHD2Q2W1FCV', 
    #    'A3DTX4Z9Z8FBVC', # TODO
    #    'AJKHQUPAKCEE6', 'A4D99Y82KOLC8',
       ]
worker_ranking = ['A13EXIOBU0UVX4', 'A11BS3GFM9T4FJ', 'AC95JAUAM2L2Z', 'A2HM35CWB7IIFM', 'A13XNZBCRSEGGY', 'A1CN4UXQNL22OO', 'ARFDXQ5RCR74G', 'A29RH02TXLXLSI', 'A2NJK5F728GAQX', 'A1NF6PELRKACS9', 'A1KOG5Q6XX0DPU', 'AL0OO84JETD31', 'A7R4F8BRS8ET0']
turker_scores_path = 'toxex/mturk/turker_mean_scores.csv'


def group(x, agg_level):
    if agg_level == 5:
        return int(x * 4 + 1)
    if agg_level == 3:
        return 1 if x > 0.67 else 0.5 if x > 0.33 else 0
    if agg_level == 2:
        return 1 if x > 0.5 else 0
    if agg_level == 6:
        return 1 if x == 1 else 0
    raise ValueError


def parse_prompt(row):
    prompt = row['prompt']
    prompt_type = row['prompt_type']
    if not np.isnan(prompt_type):
        return prompt_type
    if prompt.startswith('Explain in a sentence why the following text is offensive:'):
        return 1.0
    if prompt.startswith('Explain why the following texts are offensive.'):
        return 3.0
    return 2.0


def fleiss_kappa(M):
    """
    See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
    :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of 
        subjects and `k` is the number of categories into which assignments are made. 
        `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
    :type M: numpy matrix
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators

    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)

    kappa = (Pbar - PbarE) / (1 - PbarE)

    return kappa


def weighted_fleiss_kappa(data):
    ''' Takes the arithmetic mean of the pairwise Cohen Kappa scores between the annotations.
        This allows passing in weights so it deals with the 5-point Likert scale better
        and is more comparable to Krippendorf's alpha.
    '''
    data = (data * 4).astype(int) + 1
    combs = combinations(data.columns, 2)
    scores = [cohen_kappa_score(data[x], data[y], weights='quadratic') for x,y in combs]
    return np.mean(scores)


def add_prefix(lst, prefix):
    return [prefix + l for l in lst]


def load_data(annot_round, filter_workers=False):

    samples, output, hits_raw = [],[],[]
    if 1 in annot_round:
        samples_file = '/data2/ag/home/mturk/round_1/samples/2022-06-07-10-36-32-proc-withannots-3.csv'
        output_file = '/data2/ag/home/mturk/round_1/output_files/combined.csv'
        hits_file = '/data2/ag/home/mturk/experiment_files_round_1/annotation_results.csv'

        samples_rows = pd.read_csv(samples_file)
        output_rows = pd.read_csv(output_file)
        hits = pd.read_csv(hits_file)
        samples_rows['mturk_round'] = 1
        output_rows['mturk_round'] = 1
        hits['mturk_round'] = 1
        samples.append(samples_rows)
        output.append(output_rows)
        hits_raw.append(hits)

    if 2 in annot_round:
        samples_file = '/data2/ag/home/mturk/round_2/samples/2022-06-20_01-36-proc-withannots.csv'
        output_file = '/data2/ag/home/mturk/round_2/output_files/hits_output_20_06_22_093147.csv'
        hits_file = '/data2/ag/home/mturk/experiment_files_round_2/annotation_results.csv'

        samples_rows = pd.read_csv(samples_file)
        output_rows = pd.read_csv(output_file)
        hits = pd.read_csv(hits_file)
        samples_rows['mturk_round'] = 2
        output_rows['mturk_round'] = 2
        hits['mturk_round'] = 2
        samples.append(samples_rows)
        output.append(output_rows)
        hits_raw.append(hits)

    if 3 in annot_round:
        samples_file = '/data2/ag/home/mturk/round_3/samples/2022-06-20_01-36-proc-withannots.csv'
        output_file = '/data2/ag/home/mturk/round_3/output_files/hits_output_20_06_22_093325.csv'
        hits_file = '/data2/ag/home/mturk/experiment_files_round_3/annotation_results.csv'

        samples_rows = pd.read_csv(samples_file)
        output_rows = pd.read_csv(output_file)
        hits = pd.read_csv(hits_file)
        samples_rows['mturk_round'] = 3
        output_rows['mturk_round'] = 3
        hits['mturk_round'] = 3
        samples.append(samples_rows)
        output.append(output_rows)
        hits_raw.append(hits)

    df_samples = pd.concat(samples)
    df_output = pd.concat(output)
    df_hits_raw = pd.concat(hits_raw)

    # Manually parse prompt type from demos as these are not specified here
    df_samples['prompt_type'] = df_samples[['prompt_type','prompt']].apply(parse_prompt, axis=1)
    
    df_output.drop('target_group_str', axis=1, inplace=True)

    # df_hits = process_hits(df_hits_raw, filter_workers=filter_workers)
    df_hits_unag = process_hits(df_hits_raw, filter_workers=filter_workers)
    df_samples['target_group_str'].fillna('other', inplace=True)

    # We don't have a unique row Id (my bad :/). Only one instance of duplicates so drop this
    df_samples.drop_duplicates(['id','explanation','prompt'], inplace=True)
    df_output.drop_duplicates(['sample_id','explanation','prompt','mturk_round'], inplace=True)
 
    # # Merge hits to output file
    # df_output_agg = pd.merge(df_output, df_hits, left_on='hit_id', right_on='HITId', how='left')

    # # Obtain annotated hits with gold annotations
    # df_answers = pd.merge(
    #     df_samples, 
    #     df_output_agg.drop('dataset', axis=1),
    #     left_on=['id','explanation','prompt'], 
    #     right_on=['sample_id','explanation','prompt'],
    #     how='left'
    # )
    # df_answers = df_answers.loc[:,~df_answers.columns.duplicated()].copy()
    # df_answers['uid'] = df_answers['id'] + '::' + df_answers['prompt_type'].astype(str)
    # # df_answers = df_answers[~df_answers['annotator_1'].isin(df_answers['HITId'])]

    if False:
        # Reallocate prompts
        type_1_prompt = 'Explain in a sentence why the following text is offensive:'
        df_answers['prompt_type'][(df_answers['prompt_type']==2) & \
            df_answers['prompt'].str.startswith(type_1_prompt)] = 1

    # # Filter unaggregated rows for samples with at least 3 annotations
    # df_hits_unag = df_hits_unag[df_hits_unag['HITId'].isin(df_answers['HITId'])]

    # Merge hits to output file
    df_output_unag = pd.merge(df_hits_unag, df_output, right_on='hit_id', left_on='HITId', how='left')

    # Obtain annotated hits with gold annotations
    df_answers_unag = pd.merge(
        df_samples, 
        df_output_unag.drop('dataset', axis=1), 
        left_on=['id','explanation','prompt'], 
        right_on=['sample_id','explanation','prompt'],
        how='left'
    )
    df_answers_unag = df_answers_unag.loc[:,~df_answers_unag.columns.duplicated()].copy()
    df_answers_unag['uid'] = df_answers_unag['id'] + '::' + df_answers_unag['prompt_type'].astype(str)

    df_answers_unag = df_answers_unag.drop_duplicates(['WorkerId','uid'])

    # Filter so that only the best 3 annotators responses are left
    df_answers_unag['WorkerRank'] = df_answers_unag['WorkerId'].apply(
        lambda x: worker_ranking.index(x) if x in worker_ranking else 100)
    df_answers_unag = df_answers_unag.sort_values(['id','WorkerRank']).reset_index(drop=True)
    df_answers_unag = df_answers_unag.groupby(['id','WorkerId']).head(3)

    # df_answers_unag = normalize_turker_scores(df_answers_unag)
    
    df_answers = instance_aggregate(df_answers_unag)

    return df_samples, df_output, None, df_hits_unag, df_answers, df_answers_unag


def normalize_turker_scores(df_answers_unag):
    ''' Perform z-normalization of the turker scores to account
        for some annotators being more positive than others.
        Based on https://aclanthology.org/W13-2305.pdf
    '''
    # raise NotImplementedError("Doesn't make sense for likert scale so disable")
    turker_scores = pd.read_csv(turker_scores_path)
    df_answers_unag = pd.merge(
        df_answers_unag, 
        turker_scores[['WorkerId','annotator_mean','annotator_std']],
        on='WorkerId', 
        how='left'
    )
    # z normalize
    df_answers_unag['z_annotator_0'] = (df_answers_unag['numeric_annotator_0'] - df_answers_unag['annotator_mean']
        ) / df_answers_unag['annotator_std']
    
    def to_likert(x):
        if x < df_answers_unag['z_annotator_0'].quantile(0.2):
            return 0.0
        if x < df_answers_unag['z_annotator_0'].quantile(0.4):
            return 0.25
        if x < df_answers_unag['z_annotator_0'].quantile(0.6):
            return 0.5
        if x < df_answers_unag['z_annotator_0'].quantile(0.8):
            return 0.75
        if x < df_answers_unag['z_annotator_0'].quantile(1.):
            return 1.0

    # Remap z scores to likert scale
    df_answers_unag['znorm_numeric_annotator_0'] = df_answers_unag['z_annotator_0'].apply(to_likert)

    return df_answers_unag


def sort_by_workers(row):
    ''' Helper to order the answers for each HIT such that
        the best workers appear first.
    '''
    answers = row['answer']
    wids = row['WorkerIds']
    # Create worker ordering
    ixs = [worker_ranking.index(wid) if wid in worker_ranking else 100 for wid in wids] 
    # Sort answers and workerids by indexes
    _, sorted_wids, sorted_answers = zip(*sorted([(i,w,a) for i,a,w in zip(ixs, answers, wids)]))
    return sorted_answers, sorted_wids


def process_hits(df_hits_raw, filter_workers=False):
    # Aggregate hits
    df_hits_raw['answer'] = df_hits_raw['Answer'].apply(json.loads)
    df_hits_raw['AcceptTime'] = df_hits_raw['AcceptTime'].apply(pd.to_datetime)
    df_hits_raw['SubmitTime'] = df_hits_raw['SubmitTime'].apply(pd.to_datetime)

    if filter_workers:
        df_hits_raw = df_hits_raw[df_hits_raw['WorkerId'].isin(retain_worker_ids)]

    df_hits = df_hits_raw[['HITId','WorkerId','comment_id','answer','AcceptTime','SubmitTime','mturk_round']].groupby(
        ['HITId','WorkerId','comment_id','AcceptTime','SubmitTime','mturk_round']).aggregate(lambda lst: [x[0] for x in lst])
        
    if False:
        # Keep only best 3 annotators by row
        df_hits['worker_rankings'] = [
            worker_ranking.index(x) if x in worker_ranking else 100 
            for x in df_hits.index.get_level_values('WorkerId')
        ]
        # Sort data so the best annotations for each HIT appear first
        df_hits = df_hits.sort_values(['HITId','worker_rankings'])
        # Keep only 3 annotations per sample
        df_hits = df_hits.groupby('HITId').head(3)

        # Keep only rows with 3 or more annotations
        mask = df_hits.groupby('HITId').count()['answer'] >= 3
        count_3_ids = df_hits.groupby('HITId').count()[mask].index
        df_hits = df_hits[df_hits.index.get_level_values('HITId').isin(count_3_ids)]
        
    df_hits.reset_index(drop=False, inplace=True)

    # Convert answer dict to columns
    rows = []
    for row in df_hits.to_dict(orient='records'):
        answer = row.pop('answer')
        for i in range(len(answer)):
            row_answers = sorted(answer[i]['Answer'].items(), key = lambda x: x[1], reverse=True)
            row[f'annotator_{i}'] = row_answers[0][0]
        rows.append(row)
    df_hits = pd.DataFrame.from_dict(rows)
    # df_hits.dropna(axis=0, inplace=True)
    
    # Map to numeric
    prefix = 'numeric_'
    for c in answer_cols:
        if c not in df_hits.columns:
            continue
        df_hits[prefix+c] = df_hits[c].apply(lambda x: label_map[x] if x in label_map else x)
    numeric_cols = add_prefix([c for c in answer_cols if c in df_hits.columns], prefix)

    return df_hits


def instance_aggregate(df_answers_unag):
    
    df_answers = df_answers_unag[['id','prompt_type','numeric_annotator_0','dataset','text']]
    df_answers['annot_instance'] = df_answers.groupby(['id','prompt_type','dataset','text']).cumcount()
    df_answers = df_answers.pivot(index=['id','prompt_type','dataset','text'],columns='annot_instance',values='numeric_annotator_0')
    cols = df_answers.columns

    # Take majority vote
    df_answers['majority'] = df_answers.mode(axis='columns').iloc[:,0]
    df_answers['majority_numeric'] = df_answers['majority'].apply(lambda x: label_map[x] if x in label_map else x)

    # Take median
    df_answers['median_annotator'] = df_answers[cols].median(axis=1)
    
    # Take numeric mean
    df_answers['mean_numeric_annotator'] = df_answers[cols].mean(axis=1)

    return df_answers


def old_agg():
    # Aggregate hits
    df_hits_raw['answer'] = df_hits_raw['Answer'].apply(json.loads)
    df_hits_raw['AcceptTime'] = df_hits_raw['AcceptTime'].apply(pd.to_datetime)
    df_hits_raw['SubmitTime'] = df_hits_raw['SubmitTime'].apply(pd.to_datetime)

    if filter_workers:
        df_hits_raw = df_hits_raw[df_hits_raw['WorkerId'].isin(retain_worker_ids)]

    if aggregate:
        # Aggregate across all workers so one row per sample
        df_hits = df_hits_raw[['HITId','comment_id','answer','WorkerId']].groupby(
            ['HITId','comment_id']).aggregate({
                'answer': lambda lst: [x[0] for x in lst],
                'WorkerId': lambda x: x.tolist(),
            }).rename(columns={'WorkerId':'WorkerIds'})
        # Sort values so the best workers appear first
        assert (df_hits['answer'].apply(len) == df_hits['WorkerIds'].apply(len)).all() 
        df_hits['answer'], df_hits['WorkerIds'] = zip(
            *df_hits[['answer','WorkerIds']].apply(sort_by_workers, axis=1))

    else:
        # Don't aggregate over rows
        df_hits = df_hits_raw[['HITId','WorkerId','comment_id','answer','AcceptTime','SubmitTime']].groupby(
            ['HITId','WorkerId','comment_id','AcceptTime','SubmitTime']).aggregate(lambda lst: [x[0] for x in lst])
        
        if False:
            # Keep only best 3 annotators by row
            df_hits['worker_rankings'] = [
                worker_ranking.index(x) if x in worker_ranking else 100 
                for x in df_hits.index.get_level_values('WorkerId')
            ]
            # Sort data so the best annotations for each HIT appear first
            df_hits = df_hits.sort_values(['HITId','worker_rankings'])
            # Keep only 3 annotations per sample
            df_hits = df_hits.groupby('HITId').head(3)

            # Keep only rows with 3 or more annotations
            mask = df_hits.groupby('HITId').count()['answer'] >= 3
            count_3_ids = df_hits.groupby('HITId').count()[mask].index
            df_hits = df_hits[df_hits.index.get_level_values('HITId').isin(count_3_ids)]
        
    df_hits.reset_index(drop=False, inplace=True)

    # Convert answer dict to columns
    rows = []
    for row in df_hits.to_dict(orient='records'):
        answer = row.pop('answer')
        for i in range(len(answer)):
            row_answers = sorted(answer[i]['Answer'].items(), key = lambda x: x[1], reverse=True)
            row[f'annotator_{i}'] = row_answers[0][0]
        rows.append(row)
    df_hits = pd.DataFrame.from_dict(rows)
    # df_hits.dropna(axis=0, inplace=True)
    
    # Map to numeric
    prefix = 'numeric_'
    for c in answer_cols:
        if c not in df_hits.columns:
            continue
        df_hits[prefix+c] = df_hits[c].apply(lambda x: label_map[x] if x in label_map else x)
    numeric_cols = add_prefix([c for c in answer_cols if c in df_hits.columns], prefix)

    if not aggregate:
        return df_hits

    ## Aggregate annotator scores by sample

    # Take majority vote
    # Currently I ignore rows which have a split decision
    # TODO: find a better solution here
    majority = df_hits[[a for a in answer_cols if a in df_hits.columns]].mode(axis='columns')
    majority[~majority.iloc[:,1].isnull()] = np.nan
    df_hits['majority'] = majority.iloc[:,0]
    df_hits['majority_numeric'] = df_hits['majority'].apply(lambda x: label_map[x] if x in label_map else x)

    # Take median
    df_hits['median_annotator'] = df_hits[numeric_cols].median(axis=1)
    
    # Take numeric mean
    df_hits['mean_numeric_annotator'] = df_hits[numeric_cols].mean(axis=1)

    return df_hits


def dt_now():
    return datetime.today().strftime('%Y-%m-%d-%H-%M-%S')


if __name__ == '__main__':
    missing_annotations()
