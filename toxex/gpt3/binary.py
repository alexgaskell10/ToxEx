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

from toxex.gpt3.base import BasePredictor


class BinaryPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, text: Optional[str], **kwargs) -> dict:
        result = {'explanation': None, 'prompt': None, 'target_group_str': None, 'toxicity_type': None, 'id': id, 'demos': None}

        prompt_base = self._prompt_base
        instruct = prompt_base
        prompt_str = instruct + ':\n\n ' + text 

        # Call the gpt-3 API
        gen_dict = self._gen_dict
        gen_dict['prompt'] = prompt_str
        response = self.call_openai(gen_dict, dry_run=True)

        result['prompt'] = prompt_str
        result['target_group_str'] = None
        result['toxicity_type'] = None
        result['demos'] = None
        result['explanation'] = response['text_response'].strip('\n')

        return result