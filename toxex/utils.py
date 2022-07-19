import random
from typing import Optional, Union, List
from pathlib import Path
import torch
from torch import nn
import numpy as np
import pandas as pd
import jsonlines
import tqdm
from datetime import datetime
from string import punctuation


def dt_now():
    return datetime.today().strftime('%Y-%m-%d-%H-%M-%S')


def format_prompt(prompt):
    ''' Ensure prompts end with ".\n". This prevents the 
        completion from continuing the prompt rather than
        responding to it.
    '''
    prompt = prompt.strip()
    if not prompt[-1] in punctuation:
        prompt += '.'
    prompt += '\n'
    return  prompt


def set_seed(seed, n_gpu):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if n_gpu > 0: 
        torch.cuda.manual_seed_all(seed) 


def write_jsonl_into_file(data: List[dict], fname: str, 
                        mode: str="w", do_tqdm: bool = True) -> None:
    """Writes a list of dictionaries into the given file path as .jsonl."""
    iterator = tqdm.tqdm(data, unit="line") if do_tqdm else data
    with jsonlines.open(str(fname), mode=mode) as f:
        for line in iterator:
            f.write(line)


def select_demos(tgt_group, toxicity_type, df_seed, n_demos=3):
    ''' Selection some samples from the seed group to act as demonstrations
        within the prompt.
    '''
    # First match on target group and return if there are enough rows
    rows = df_seed[df_seed['target_group'] == tgt_group]
    if len(rows) >= n_demos:
        return rows.sample(3)

    # If not enough in target group, match on toxicity type
    other_rows = df_seed[(df_seed['toxicity_type'] == toxicity_type) & ~df_seed['id'].isin(rows['id'])]
    n_other_samples = min(n_demos - len(rows), len(other_rows))
    rows = rows.append(other_rows.sample(n_other_samples))
    if len(rows) == n_demos:
        return rows.iloc[::-1]      # Reverse ordering as latter demos are more influential
    
    # If not enough in toxicity type, supplement with non-matched rows
    final_rows = df_seed[~df_seed['id'].isin(rows['id'])]
    rows = rows.append(final_rows.sample(n_demos - len(rows)))
    return rows.iloc[::-1] 

 
def select_demos_group_dataset(criterion, toxicity_type, df_seed, n_demos=3):
    ''' Selection some samples from the seed group to act as demonstrations
        within the prompt.
    '''
    # First match on target group and return if there are enough rows
    rows = df_seed[(df_seed['target_group'] == criterion[0]) & (df_seed['dataset'] == criterion[1])]
    if len(rows) >= n_demos:
        return rows.sample(n_demos), 1

    # If not enough in target group + dataset, match on target group only
    other_rows = df_seed[(df_seed['target_group'] == criterion[0]) & ~df_seed['id'].isin(rows['id'])]
    n_other_samples = min(n_demos - len(rows), len(other_rows))
    rows = rows.append(other_rows.sample(n_other_samples))
    if len(rows) == n_demos:
        return rows.iloc[::-1], 2      # Reverse ordering as latter demos are more influential apparently

    # If not enough in target group, match on toxicity type
    other_rows = df_seed[(df_seed['toxicity_type'] == toxicity_type) & ~df_seed['id'].isin(rows['id'])]
    n_other_samples = min(n_demos - len(rows), len(other_rows))
    rows = rows.append(other_rows.sample(n_other_samples))
    if len(rows) == n_demos:
        return rows.iloc[::-1], 3      # Reverse ordering as latter demos are more influential apparently
    
    # If not enough in toxicity type, supplement with non-matched rows
    final_rows = df_seed[~df_seed['id'].isin(rows['id'])]
    rows = rows.append(final_rows.sample(n_demos - len(rows)))
    return rows.iloc[::-1], 4 


def vec_to_target_name_and_toxicity_type(vec: List[int], target_map: List[str], student_map: dict):
    """ Convert the target group vector ([0,1,1,0,...]) into
        the fields required to generate the prompt.

        return:
        - target_mentions: list of strings containing the names
        of the target groups in the text (e.g. ['black people', 'Muslims'])
        - toxicity_types: list of strings containing the toxicity
        type corresponding to the target groups in cat_lst
        (e.g. ['racist', 'Islamophobic'])
    """
    if all(v==-1 for v in vec) or all(v==0 for v in vec):
        tgt_lst = ['other']
    else:
        assert len(target_map) == len(vec)
        # Map to category names
        tgt_lst = [g for v,g in zip(vec, target_map) if v]
    # Map to coarser labels for gpt-3 explainability
    target_mentions = [student_map['target_group_to_name'][l] for l in tgt_lst]
    # Obtain toxicity type
    toxicity_types = [student_map['name_to_toxicity_type'][l] for l in target_mentions]
    return target_mentions, toxicity_types


def read_jsonl(fname, fields=None):
    """Reads a given .jsonl file for inference."""
    import jsonlines, tqdm
    data = []
    with jsonlines.open(fname) as f:
        for line in tqdm.tqdm(f, unit='line'):
            if fields:
                data.append({k: line[k] for k in fields})
            else:
                data.append(line)
    return data


def load_data(
    fnames: dict, 
    fields: list = ['id', 'text', 'toxicity', 'target_group'], 
    max_samples: int = 10, 
    exclude_files: list = None, 
    include_files: list = None, 
    require_target_group: bool =False
):
    ''' Function to load data from jsonl files. This includes functionality
        to include / exclude sample ids.

        :args:
        - fnames: dict where the keys are the dataset prefix and the values
        are jsonl files
        - fields: list containing the string field names to load from jsonl
        - max samples: the number of samples to return. Set to -1 to return all.
        - include_files: sample ids within those from fnames that we want to
        generate responses for. This is a dict where the keys are the csv file paths 
        and the values are the corresponding fieldname of the ID field
        - exclude_files: sample ids within those from fnames that DO NOT want to
        generate responses for (e.g. to prevent duplicate generations). This is a 
        dict where the keys are the jsonl file paths and the values are the 
        corresponding fieldname of the ID field
        - require_target_group: bool specifiying whether we want to filter out 
        samples which do not have a gold target group

        :returns:
        - data: list of dicts
    '''
    if exclude_files:
        # How to filter out samples from generation based on their id
        # (NOTE: these should include the relevant dataset prefix)
        exclude_ids = []
        for file, id_col in exclude_files.items():
            exclude_ids.extend(pd.read_json(file, orient='records', lines=True)[id_col].tolist())

    if include_files:
        # How to specify a set of sample ids to include 
        # (NOTE: these should include the relevant dataset prefix)
        include_ids = []
        for file, id_col in include_files.items():
            include_ids.extend(pd.read_csv(file)[id_col].drop_duplicates().tolist())

    # Load data
    data = []
    for dset, fname in fnames.items():
        rows = [{**row, 'dataset': dset} for row in read_jsonl(fname, fields)]

        if require_target_group:
            # Exclude samples with no target group
            rows = [row for row in rows if not all(i==-1 for i in row['target_group'])]

        # Add dataset prefix to id to make a UID
        rows = [{**row, 'id': dset + '-' + str(row['id'])} for row in rows]

        if exclude_files:
            # Exclude rows from seed set by matching on ids
            rows = [r for r in rows if r['id'] not in exclude_ids]

        if include_files:
            rows = [r for r in rows if r['id'] in include_ids]
        else:
            random.shuffle(rows)
            if max_samples > 0:
                rows = rows[:max_samples]
        
        data.extend(rows)

    return data


def read_txt_lines(fname):
    return [x.rstrip('\n') for x in open(fname).readlines()]