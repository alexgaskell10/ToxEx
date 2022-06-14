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


def read_jsonl(fname, fields):
    """Reads a given .jsonl file for inference."""
    import jsonlines, tqdm
    data = []
    with jsonlines.open(fname) as f:
        for line in tqdm.tqdm(f, unit='line'):
            data.append({k: line[k] for k in fields})
    return data
