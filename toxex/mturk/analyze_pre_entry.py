import random
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from .common import (
    group,
)



np.random.seed(1)
random.seed(1)

label_map = {
    'Strongly disagree': 0.0,
    'Weakly disagree': 0.25,
    'Neutral': 0.5,
    'Weakly agree': 0.75,
    'Strongly agree': 1.0
}
filter_wids = ['ABCD', 'A']

samples_path = '/data2/ag/home/mturk/pre-entry-test/samples/2022-06-08-21-33-47.csv'
responses_path = '/data2/ag/home/mturk/pre-entry-test/responses/Entry Test - Explanation Evaluation (Responses) - Form responses 1.csv'
outfile = '/data2/ag/home/mturk/_qualifications/create_preapproved_workers/prelim_turker_ids.txt'


def load_pre_entry():
    # Load ground truths
    df_samples = pd.read_csv(samples_path)
    df_samples['qid'] = df_samples.index
    df_samples = df_samples.rename(columns={'avg_response': 'label'})
    qid_scores = df_samples[['qid', 'label']]

    # Load survey responses
    df_responses = pd.read_csv(responses_path)
    df_responses.columns = ['Timestamp', 'Email address', *range(20), 'Worker ID']
    df_responses = df_responses.melt(id_vars=['Timestamp', 'Email address', 'Worker ID'])
    df_responses = df_responses.rename(columns={'variable': 'qid'})
    df_responses['answer'] = df_responses['value'].apply(lambda x: label_map[x])

    # Merge labels
    df_responses = pd.merge(df_responses, qid_scores, on='qid')

    return df_responses


def pre_entry_performance_by_acceptance(df_responses, agg_level, retain_worker_ids):
    data = df_responses[['Worker ID','label','answer']].dropna()
    data['accepted_worker'] = data['Worker ID'].isin(retain_worker_ids)
    data['gold'] = data['label'].apply(lambda x: group(x, agg_level))
    data['pred'] = data['answer'].apply(lambda x: group(x, agg_level))
    data['correct'] = data['pred'] == data['gold']
    return {
        f'all-{agg_level}': data['correct'][data['accepted_worker']].mean(),
        f'passed-{agg_level}': data['correct'][data['accepted_worker']].mean(),
        f'failed-{agg_level}': data['correct'][~data['accepted_worker']].mean(),
    }


def compute_pre_entry_performance_by_acceptance():
    df_responses = load_pre_entry()

    retain_worker_ids_pre_entry = [l.strip('\n') for l in open(outfile, 'r').readlines()]

    results = {}
    for agg_level in [2,3]:
        results.update(pre_entry_performance_by_acceptance(
            df_responses, agg_level, retain_worker_ids_pre_entry))
    
    return {'preentry_'+k:v for k,v in results.items()}
