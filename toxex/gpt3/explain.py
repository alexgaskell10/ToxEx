#!/usr/bin/env python

import os
import json
import logging
import random

from collections import namedtuple
from typing import Optional, List

import numpy as np

import torch
import openai

import time
import pandas as pd
from string import punctuation

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


set_seed(1, 1)

Result = namedtuple(
    'Result', 'explanation prompt target_group toxicity logprobs response'
)


class Explainer:
    def __init__(
        self,
        gpt_tgt_group_mapping_fname: str,
        gpt_opts: dict,
        api_key: Optional[str] = None,
        target_group_threshold: Optional[float] = 0.5,
        demos_fname: Optional[str] = None,
        demos_threshold: Optional[float] = 0.8,
        dry_run: bool = True,
        prompt_type: str = None,
    ):
        self._gpt_tgt_group_mapping = json.load(open(gpt_tgt_group_mapping_fname, 'r'))
        self._gpt_opts = gpt_opts
        self._api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self._dry_run = dry_run
        self._tgt_group_thresh = target_group_threshold
        self._prompt_type = prompt_type

        if self._prompt_type == 3:
            self.build_prompt = self._build_prompt_with_demos
            self._demos_thresh = demos_threshold
            self._demos = self._load_demos(demos_fname, self._demos_thresh)
            self._prompt_preamble = 'Explain why the following texts are offensive.'
            self._prompt_base = 'Explain in a sentence why the following text is <TOXICITY_TYPE> to'
            self._n_demos = 3
        elif self._prompt_type == 2:
            self.build_prompt = self._build_prompt_tgt
            self._prompt_base = 'Explain in a sentence why the following text is <TOXICITY_TYPE> to'
        elif self._prompt_type == 1:
            self.build_prompt = self._build_prompt
            self._prompt_base = 'Explain in a sentence why the following text is offensive:'

    def predict(
            self, id: str, text: str, toxicity: int, tgt_groups_names: List[str], tgt_groups_scores: List[float],
    ) -> dict:
        # Currently we only explain toxic samples
        if not toxicity:
            return Result(
                explanation=None, prompt=None, target_group=None,
                toxicity=None, logprobs=None, response=None,
            )._asdict()

        # Select the most probable target group
        # We select `other` if the most probable group has the probability lower than threshold
        idxs = np.argsort(tgt_groups_scores)[::-1]
        scores, names = list(zip(*[(tgt_groups_scores[idx], tgt_groups_names[idx]) for idx in idxs]))
        group = 'other' if len(names) == 0 or scores[0] < self._tgt_group_thresh else names[0]
        group_for_prompt = self._gpt_tgt_group_mapping['target_group_to_name'][group]
        toxicity_type = self._gpt_tgt_group_mapping['name_to_toxicity_type'][group_for_prompt]

        # Create prompt
        prompt_str, demos = self.build_prompt(text, toxicity_type, group_for_prompt)

        # Call the gpt-3 API
        self._gpt_opts['prompt'] = prompt_str
        response = self._call_openai()
        return Result(
            explanation=response['text_response'].strip('\n'),
            prompt=prompt_str,
            target_group=group_for_prompt,
            toxicity=toxicity_type,
            logprobs=response['logprobs'],
            response=response,
        )._asdict()

    def _build_prompt(self, text, *args):
        """ Build prompt with no toxicity type and target group."""
        instruct = self._prompt_base
        prompt_str = self._format_prompt(instruct + ':\n\n ' + text)
        return prompt_str, None

    def _build_prompt_tgt(self, text, toxicity_type, target_group):
        """ Build prompt with toxicity type and target group but not demos."""
        prompt_core = self._prompt_base.replace('<TOXICITY_TYPE>', toxicity_type)
        instruct = f'{prompt_core} {target_group.strip()}' if target_group != 'other' else prompt_core.rstrip(' to')
        prompt_str = self._format_prompt(instruct + ':\n\n ' + text)
        return prompt_str, None

    def _build_prompt_with_demos(self, text, toxicity_type, target_group):
        """ Build prompt with toxicity type, target group and demos."""

        # Select demos from the preloaded set and format correctly
        demos = self._select_demos(target_group, toxicity_type)
        demos = demos.reset_index(drop=True).reset_index()
        demos['demo_str'] = demos[['index', 'prompt_sample', 'text_response']].apply(
            self._make_demo_string, axis=1)
        demos_str = '\n\n'.join(demos['demo_str'].tolist())

        # Create prompt
        prompt_core = self._prompt_base.replace('<TOXICITY_TYPE>', toxicity_type)
        instruct = f'{prompt_core} {target_group.strip()}' \
            if target_group != 'other' else prompt_core.replace(' to', '')
        sample_prompt_str = f'Example {self._n_demos + 1}: {instruct}: {text}\nExplanation {self._n_demos + 1}:'
        prompt_str = '\n\n'.join([self._prompt_preamble, self._format_prompt(demos_str), sample_prompt_str])

        return prompt_str, demos_str

    def _call_openai(self):
        """ Calls the OpenAI model by sending a prompt and saving the prompt-response pair in an output file,
            may also updating the prompt file for continued interaction with the chatbot
        """
        # Load your API key
        openai.api_key = self._api_key

        st = time.time()
        if self._dry_run:
            response = {'choices': [{'text': 'abc'}]}
        else:
            print('** Calling OpenAI\'s API.... **')
            response = openai.Completion.create(**self._gpt_opts)
        dur = time.time() - st

        # Parse output
        jdata = dict()
        jdata['response'] = response
        jdata['text_response'] = response['choices'][0]['text'] if len(response['choices']) == 1 else None
        jdata['response_time'] = dur
        jdata['prompt'] = self._gpt_opts.pop('prompt')
        jdata['gpt_opts'] = self._gpt_opts
        jdata['logprobs'] = response['choices'][0]['logprobs']['token_logprobs'][:len(jdata['text_response'].split())+2] \
            if len(response['choices']) == 1 and not self._dry_run else None

        return jdata

    def _select_demos(self, target_group, toxicity_type):
        """ Select some samples from the seed group to act as demonstrations
            within the prompt.
        """

        # First match on target group and return if there are enough rows
        rows = self._demos[self._demos['target_group'] == target_group]
        if len(rows) >= self._n_demos:
            return rows.sample(3)

        # If not enough in target group, match on toxicity type
        other_rows = self._demos[(self._demos['toxicity_type'] == toxicity_type) & \
                                 ~self._demos['id'].isin(rows['id'])]
        n_other_samples = min(self._n_demos - len(rows), len(other_rows))
        rows = rows.append(other_rows.sample(n_other_samples))
        if len(rows) == self._n_demos:
            return rows.iloc[::-1]  # Reverse ordering as latter demos are more influential

        # If not enough in toxicity type, supplement with non-matched rows
        final_rows = self._demos[~self._demos['id'].isin(rows['id'])]
        rows = rows.append(final_rows.sample(self._n_demos - len(rows)))
        return rows.iloc[::-1]

    @staticmethod
    def _load_demos(fname: str, threshold: float):
        """ This method ingests a path to the set of annotated gpt-3 generations.
            These will be loaded and filtered for highly scoring ones.

            threshold: reject seed examples with avg annotation score < threshold
        """
        df = pd.read_json(fname, orient='records', lines=True)
        df = df[df['avg_response'] >= threshold]
        return df

    @staticmethod
    def _format_prompt(prompt):
        """ Ensure prompts end with ".\n". This prevents the
            completion from continuing the prompt rather than
            responding to it.
        """
        prompt = prompt.strip()
        if not prompt[-1] in punctuation:
            prompt += '.'
        prompt += '\n'
        return prompt

    @staticmethod
    def _make_demo_string(x):
        """ Helper to create prompt-response strings from demos as follows:

            Example 1: Explain in a sentence why the following text is offensive to LGBTQ+
                people: trans like to be as obnoxious as possible.
            Explanation 1: The text is offensive because it perpetuates the idea that
                transgender people are only interested in causing trouble and being
                disruptive. This is not only untrue, but it also contributes to the
                discrimination and mistreatment that transgender people often face.
        """
        n = x['index']
        prompt_sample = x['prompt_sample']
        text_response = x['text_response']
        return f'Example {n + 1}: {prompt_sample}\nExplanation {n + 1}: {text_response}'