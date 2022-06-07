#!/usr/bin/env python
import os
import json
import random
from typing import Optional, Union, List
from pathlib import Path
from unittest.mock import Base
from torch import nn
import numpy as np
import pandas as pd

import torch
import openai

import time

from toxex.utils import (
    read_jsonl,
    set_seed,
    vec_to_target_name_and_toxicity_type, 
    select_demos_group_dataset,
    select_demos
)
from toxex.gpt3.base import BasePredictor


class Explainer(BasePredictor):
    def __init__(self, 
        *args, 
        demos_file: str = None, 
        target_group_map_file: str = None, 
        target_groups: str = None, 
        prompt_type: int = None,
        demos_threshold: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._target_group_map_file = target_group_map_file
        self._target_groups = target_groups
        self._demos_file = demos_file
        self._use_demos = demos_file is not None
        self._prompt_type = prompt_type

        if self._prompt_type == 1:
            self.build_prompt = self._build_type1_prompt
        else:
            self._tgt_map = self._load_target_group_map(self._target_group_map_file)
            assert all(x == y for x,y in zip(
                sorted(self._target_groups), sorted(self._tgt_map['target_group_to_name'])))

        if self._prompt_type == 2:
            self.build_prompt = self._build_type2_prompt
        elif self._prompt_type == 3:
            self.build_prompt = self._build_type3_prompt
            self._thd = demos_threshold
            self._demos = self._load_demos(self._demos_file, self._thd)
            self._prompt_preamble = 'Explain why the following texts are offensive.'
            self._n_demos = 3

    def _load_demos(self, fname: str, thd: float):
        """ This method ingests a path to the set of annotated gpt-3 generations.
            These will be loaded and filtered for highly scoring ones.

            thd: reject seed examples with avg annotation score < threshold
        """
        df = pd.read_json(fname, orient='records', lines=True)        # TODO: maybe use something other than padas here
        df = df[df['avg_response'] >= thd]
        return df
    
    def _load_target_group_map(self, fname: str) -> dict:
        """ This method ingests a filename and returns a dict
            for mapping provided target group into a string name.
        """
        return json.load(open(fname, 'r'))

    def predict(self, text: Optional[str], target_group: List[int], gen_dict: dict,
                toxicity: int, id: str = None, dataset: str = None, dry_run: bool = True,
                **kwargs) -> dict:

        result = {
            'explanation': None,
            'prompt': None,
            'target_group_str': None, 
            'toxicity_type': None,
            'id': id,
            'demos': None,
            'dataset': dataset,
            'prompt_type': self._prompt_type,
        }

        if not toxicity:
            # Currently we only explain toxic samples
            return result

        prompt_str, tgt_group, toxicity_type, demos = self.build_prompt(
            text, target_group=target_group, dataset=dataset)
        
        # Call the gpt-3 API
        gen_dict['prompt'] = prompt_str
        response = self.call_openai(gen_dict, dry_run=dry_run)

        result['prompt'] = prompt_str
        result['target_group_str'] = tgt_group
        result['toxicity_type'] = toxicity_type
        result['demos'] = demos
        result['explanation'] = response['text_response'].strip('\n')

        # For backward compatability
        result['text_response'] = result['explanation']

        return result

    def _build_type1_prompt(self, text, **kwargs):
        ''' Without toxicity type, target group and demos.
        '''
        tgt_group = None
        toxicity_type = None

        # Create prompt
        prompt_base = self._prompt_base
        prompt_core = prompt_base.replace('<TOXICITY_TYPE>', 'offensive')
        instruct = prompt_core.rstrip(' to')
        prompt_str = instruct + ':\n\n ' + text
        demos = None

        return prompt_str, tgt_group, toxicity_type, demos

    def _build_type2_prompt(self, text, target_group, **kwargs):
        ''' With toxicity type and target group but not demos.
        '''
        # Convert from target group binary vector to a list of the string mentions
        target_mentions, toxicity_types = vec_to_target_name_and_toxicity_type(
            target_group, self._target_groups, self._tgt_map)

        # Select toxicity type and target group from the available options
        ix = random.choice(range(len(target_mentions)))
        tgt_group = target_mentions[ix]
        toxicity_type = toxicity_types[ix]

        # Create prompt
        prompt_base = self._prompt_base
        prompt_core = prompt_base.replace('<TOXICITY_TYPE>', toxicity_type)
        instruct = f'{prompt_core} {tgt_group.strip()}' if tgt_group != 'other' else prompt_core.rstrip(' to')
        prompt_str = instruct + ':\n\n ' + text
        demos = None

        return prompt_str, tgt_group, toxicity_type, demos

    def _build_type3_prompt(self, text, target_group, dataset):
        ''' With toxicity type, target group and demos.
        '''
        # Convert from target group binary vector to a list of the string mentions
        target_mentions, toxicity_types = vec_to_target_name_and_toxicity_type(
            target_group, self._target_groups, self._tgt_map)

        # Select toxicity type and target group from the available options
        ix = random.choice(range(len(target_mentions)))
        tgt_group = target_mentions[ix]
        toxicity_type = toxicity_types[ix]

        if dataset is not None:
            # If we have the dataset, select demos favoring those from the same dataset
            criterion = (tgt_group, dataset)
            demos, _ = select_demos_group_dataset(criterion, toxicity_type, self._demos, self._n_demos)
        else:
            demos = select_demos(tgt_group, toxicity_type, self._demos, self._n_demos)

        # Create demo string
        demos['demo_str'] = ''
        for n,demo in enumerate(demos.to_dict(orient='records')):
            # Create examples as follows:
            # Example 1: Explain in a sentence why the following text is offensive to LGBTQ+ people: trans like to be as obnoxious as possible.
            # Explanation 1: The text is offensive because it perpetuates the idea that transgender people are only interested in causing trouble and being disruptive. This is not only untrue, but it also contributes to the discrimination and mistreatment that transgender people often face.
            demos['demo_str'].iloc[n] = (f'Example {n+1}: {demo["prompt_sample"]}'
                                        f'\nExplanation {n+1}: {demo["text_response"]}')
        demos_str = '\n\n'.join(demos['demo_str'].tolist())

        # Create prompt
        prompt_core = self._prompt_base.replace('<TOXICITY_TYPE>', toxicity_type)
        instruct = f'{prompt_core} {tgt_group.strip()}' \
            if tgt_group != 'other' else prompt_core.rstrip(' to')
        sample_prompt_str = f'Example {self._n_demos+1}: {instruct}: {text}\nExplanation {self._n_demos+1}:'
        prompt_str = '\n\n'.join([self._prompt_preamble, demos_str, sample_prompt_str])

        assertion_checks(target_mentions, toxicity_types, demos, prompt_str, self._n_demos)

        return prompt_str, tgt_group, toxicity_type, demos_str


def assertion_checks(target_mentions, toxicity_types, demos, prompt_str, n_demos):
    assert len(target_mentions) == len(toxicity_types)
    assert len(demos['id'].unique()) == len(demos)
    assert all(prompt_str.count(f'Explanation {i+1}')==1 for i in range(n_demos))



'''
To do:
- Ability to customize which type of prompts we produce
- Load seed annotations if required (and specify threshold)
- Integrate returned result into the data row and align with previous format
'''