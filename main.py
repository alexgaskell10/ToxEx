import os
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from toxex.gpt3.binary import BinaryPredictor
from toxex.gpt3.explain import Explainer
from toxex.utils import (
    read_jsonl,
    set_seed,
    vec_to_target_name_and_toxicity_type, 
    select_demos_group_dataset,
    write_jsonl_into_file,
    dt_now,
)


def binary():
    fname = '/vol/bitbucket/aeg19/ag/datasets/data/sbf/jsonl.cxpr.splits/dev.jsonl'
    fields = ['text', 'toxicity', 'target_group']   # XXX
    prompts = [
        'Is the following text offensive? Answer yes or no.',
        'How offensive is the following text on a scale of 1 to 10?',
    ]
    prompt_base = prompts[0]
    
    api_key = os.getenv("OPENAI_API_KEY")    # XXX
    gen_dict = {        # XXX
        "engine": "text-davinci-002",
        "temperature": 0.9,
        "max_tokens": 384,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.25,
        "stop": '\\n',
        "n": 1,
    }

    data = read_jsonl(fname, fields)

    be = BinaryPredictor(
        api_key=api_key,
        gen_dict=gen_dict,
        prompt_base=prompt_base,
    )

    ress = []
    for d in data:
        res = be.predict(**d)
        ress.append(res)

    print('Done')


def explain():
    fnames = {
        'sbf': '/vol/bitbucket/aeg19/ag/datasets/data-aux/sbf/jsonl.cxpr.splits/test.jsonl',
        'unib': '/vol/bitbucket/aeg19/ag/datasets/data-aux/jigsaw-task2/jsonl.cxpr.splits/train.jsonl',
        'mhs': '/vol/bitbucket/aeg19/ag/datasets/data-aux/measuring_hate_speech/jsonl.cxpr.splits/test.jsonl',
    }
    outfile = Path(f'data/generations/{dt_now()}.jsonl')
    outfile.parent.mkdir(exist_ok=True)
    print('Output will be written to:', outfile)
    fields = ['id', 'text', 'toxicity', 'target_group']
    target_group_map_file = 'resources/student_name_map.json'
    target_groups = ['black', 'white', 'mixed_race', 'eastern_european',
                    'western_european', 'middle_eastern', 'eastern_asian',
                    'western_asian', 'other_asian', 'northern_american',
                    'hispanic', 'african', 'other_race', 'buddhism',
                    'christian', 'hindu', 'islam', 'jewish', 'atheist',
                    'other_religion', 'male', 'female', 'transgendered',
                    'other_gender', 'heterosexual', 'homosexual',
                    'other_LGBTQ', 'mentally_disabled', 'physically_disabled',
                    'intellectually_disabled', 'other_disabled', 'overweight',
                    'short', 'other_appearance', 'resettled', 'ethnic_minorities',
                    'age_related', 'financial_situation_related',
                    'politically_left_oriented', 'politically_right_oriented',
                    'other_politically_colored', 'profession_related',
                    'assault_victims', 'other', 
                    'atrocity_victims', 'foreigners', 'refugees', 'appearance_female']
    prompt_base = 'Explain in a sentence why the following text is <TOXICITY_TYPE> to'
    
    api_key = os.getenv("OPENAI_API_KEY")
    demos_file = 'data/annotations/annotations.jsonl'
    gen_dict = {
        "engine": "text-davinci-002",
        "temperature": 0.9,
        "max_tokens": 512,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.25,
        "stop": '\\n',
        "n": 1,
    }
    max_samples = 200
    demos_threshold = 0.8
    prompt_types = [1,2,3]

    # Exclude ids
    exclude_rows = pd.read_json(demos_file, orient='records', lines=True)        # TODO: maybe use something other than padas here

    # Load data
    data = []
    for dset, fname in fnames.items():
        rows = [{**row, 'dataset': dset} for row in read_jsonl(fname, fields) 
            if not all(i==-1 for i in row['target_group'])]

        # Exclude rows from seed set by matching on ids
        rows = [{**row, 'id': dset + '-' + str(row['id'])} for row in rows]
        rows = [r for r in rows if r['id'] not in exclude_rows['id']]

        random.shuffle(rows)
        if max_samples > 0:
            rows = rows[:max_samples]
        data.extend(rows)

    for prompt_type in prompt_types:
        be = Explainer(
            demos_file=demos_file,
            api_key=api_key,
            target_groups=target_groups,
            target_group_map_file=target_group_map_file,
            prompt_base=prompt_base,
            demos_threshold=demos_threshold,
            prompt_type=prompt_type,
        )

        results = []
        for d in tqdm(data):
            res = be.predict(**d, gen_dict=gen_dict, dry_run=False)
            output = {**d, **res}
            write_jsonl_into_file([output], outfile, mode='a', do_tqdm=False)
            results.append(output)

    print('Done')


if __name__ == '__main__':
    # binary()
    explain()