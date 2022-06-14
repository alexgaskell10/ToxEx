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
    select_demos,
    format_prompt
)
from toxex.gpt3.base import BasePredictor


class BinaryPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, text: Optional[str], id: str = None, 
                **kwargs) -> dict:
        
        result = {
            'id': id,
            'explanation': None,
            'prompt': None,
            'target_group_str': None,
            'toxicity_type': None,
            'demos': None,
            'prompt_base': self._prompt_base,
        }

        prompt_str = self._build_prompt(text)

        # Call the gpt-3 API
        gen_dict = self._gen_dict
        gen_dict['prompt'] = prompt_str
        response = self.call_openai(gen_dict, dry_run=self._dry_run)

        result['prompt'] = prompt_str
        result['target_group_str'] = None
        result['toxicity_type'] = None
        result['demos'] = None
        result['explanation'] = response['text_response'].strip('\n')

        return result

    def _build_prompt(self, text: str):
        return format_prompt(self._prompt_base + '\n\n ' + text)

    # def postprocess(self, file: str):