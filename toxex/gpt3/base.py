#!/usr/bin/env python
import os
import json
import random
from typing import Optional, Union, List
from pathlib import Path
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
)


class BasePredictor:
    def __init__(self, gen_dict: dict = None, prompt_base: str = None,
                api_key: str = None, dry_run: bool = True, **kwargs):
        self._gen_dict = gen_dict
        self._api_key = api_key
        self._prompt_base = prompt_base
        self._dry_run = dry_run

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def call_openai(self, gen_dict, output_fname=None, dry_run=True):
        """ Calls the OpenAI model by sending a prompt and saving the prompt-response pair in an output file,
            may also updating the prompt file for continued interaction with the chatbot
        """
        # Load your API key
        openai.api_key = self._api_key

        st = time.time()
        if dry_run:
            response = {'choices':[{'text':'abc'}]}
        else:
            print('** Calling OpenAI\'s API.... **')
            response = openai.Completion.create(**gen_dict)
        dur = time.time() - st

        # Parse output
        jdata = {}
        jdata['response'] = response
        jdata['text_response'] = response['choices'][0]['text'] if len(response['choices']) == 1 else None
        jdata['response_time'] = dur
        jdata['prompt'] = gen_dict.pop('prompt')
        jdata['gen_dict'] = gen_dict

        # save the prompt and API call response in a json file
        if output_fname:
            with open(output_fname, 'a') as output_f:
                output_f.write(json.dumps(jdata) + '\n')

        return jdata
