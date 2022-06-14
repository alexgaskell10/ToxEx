import os
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
import shutil

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


set_seed(1, 1)

api_key = os.getenv("OPENAI_API_KEY")


def load_data(fnames, fields, max_samples, exclude_files=None, require_target_group=False):
    if exclude_files:
        exclude_ids = []
        for file in exclude_files:
            # TODO: maybe use something other than padas here
            exclude_ids.extend(pd.read_json(file, orient='records', lines=True)['id'].tolist())

    # Load data
    data = []
    for dset, fname in fnames.items():
        rows = [{**row, 'dataset': dset} for row in read_jsonl(fname, fields)]

        if require_target_group:
            # Exclude samples with no target group
            rows = [row for row in rows if not all(i==-1 for i in row['target_group'])]

        # Add dataset annotation to id
        rows = [{**row, 'id': dset + '-' + str(row['id'])} for row in rows]

        if exclude_files:
            # Exclude rows from seed set by matching on ids
            rows = [r for r in rows if r['id'] not in exclude_ids]

        random.shuffle(rows)
        if max_samples > 0:
            rows = rows[:max_samples]
        data.extend(rows)
    return data


def find_match(exp, regexs):
    for reg, sub in regexs:
        search = re.search(reg, exp)
        if search:
            return re.sub(reg, sub, exp)
    return None
            

regexs_all = [
    # contains "is offensive"
    (r'.*(is offensive).*', '1'),
    # contains "not/nothing offensive"
    (r'.*((?:not|nothing) offensive).*', '0'),
    # contains "unoffensive"
    (r'.*(is unoffensive).*', '0'),
]
regexs_1 = [
    # Line starts with yes/no
    (r'(^Yes).*', '1'),
    (r'(^No).*', '0'),
]
regexs_2 = [
    # Contains a digit. 
    # TODO: find more robust solution here as will break if multiple numbers are in text
    (r'.*(\b[1-5]\b).*', '0'),
    (r'.*(\b[6-9]\b|\b10\b).*', '1'),
]
regexs_3 = [
]


def parse_answer(row, prompts):
    exp = row['explanation']
    prompt_base = row['prompt_base']
    id = row['id']

    prompt_ix = prompts.index(prompt_base)

    answer = find_match(exp, regexs_all)
    if not answer and prompt_ix == 0:
        answer = find_match(exp, regexs_1)
    if not answer and prompt_ix == 1:
        answer = find_match(exp, regexs_2)
    if not answer and prompt_ix == 2:
        answer = find_match(exp, regexs_3)

    if answer not in ['0','1']:
        # print(id, prompt_base, exp)
        return -1

    return int(answer)


def postprocess(file):

    file = Path(file)

    prompts = [
        'Is the following text toxic? Answer yes or no.',
        'On a scale of 1-10, how offensive is the following text?',
        'Is the following text offensive or unoffensive?',
    ] 
    df = pd.read_json(file, orient='records', lines=True)
    df = df.rename(columns={'toxicity': 'label'})
    df['prompt_id'] = df['prompt_base'].apply(lambda x: prompts.index(x))
    df['toxicity'] = df[['explanation', 'prompt_base', 'id']].apply(
        lambda x: parse_answer(x, prompts), axis=1)

    null_rows = df[df['toxicity'] == -1]
    if len(null_rows):
        print(f'Answer could not be parsed for {len(null_rows)} rows!')
        df = df[df['toxicity'] != -1]

    df['correct'] = df['toxicity'] == df['label']
    # df[['correct','prompt_base']].groupby('prompt_base').mean()
    # df[['prompt_base', 'correct']].value_counts()
    # print(df[['toxicity', 'explanation']].values)

    # reformat so this can be evaluated using cxpy
    col_map = {
        'toxicity': 'pred_label_id',
        'label': 'bin_label_id',      
    }
    input_cols = ['id', 'text', 'prompt_base']
    df_cxpy = df[['dataset','prompt_id']+list(col_map)].rename(columns=col_map)
    df_cxpy['input'] = df[input_cols].to_dict(orient='records')

    # Write outputs
    outdir = Path(str(file).rstrip('.jsonl'))
    print('Writing outputs to', outdir)
    outdir.mkdir(exist_ok=True)
    write_jsonl_into_file(df_cxpy.to_dict(orient='records'), Path(*file.parts[:2],file.stem,'all.jsonl'))
    for dset,pid in df_cxpy[['dataset', 'prompt_id']].drop_duplicates().values:
        outpath = outdir / f'{dset}-pid_{pid}.jsonl'
        rows = df_cxpy[(df_cxpy['dataset'] == dset) & (df_cxpy['prompt_id'] == pid)]
        write_jsonl_into_file(rows.to_dict(orient='records'), outpath)

    print("done")


def binary():
    fnames = {
        'sbf': '/data2/ag/home/ag/datasets/data/sbf/jsonl.cxpr.splits/test.jsonl',
        'unib': '/data2/ag/home/ag/datasets/data/jigsaw-task2/jsonl.cxpr.splits/test.jsonl',
        'mhs': '/data2/ag/home/ag/datasets/data/measuring_hate_speech/jsonl.cxpr.splits/test.jsonl',
    }
    outfile = Path(f'data/binary/{dt_now()}.jsonl')
    outfile.parent.mkdir(exist_ok=True)
    print('Output will be written to:', outfile)
    fields = ['id', 'text', 'toxicity', 'target_group']
    prompts = [
        'Is the following text toxic? Answer yes or no.',
        'On a scale of 1-10, how offensive is the following text?',
        'Is the following text offensive or unoffensive?',
    ] 
    gen_dict = {
        "engine": "text-davinci-002",
        "temperature": 0.0,
        "max_tokens": 10,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.25,
        "stop": '\\n',
        "n": 1,
    }
    max_samples = -1
    postprocess = True
    exclude_files = [
        '/data2/ag/home/ToxEx/data/binary/aaa-2022-06-09-22-12-58.jsonl',
    ]

    data = load_data(fnames, fields, max_samples, exclude_files=exclude_files)

    for prompt in prompts:
        be = BinaryPredictor(
            api_key=api_key,
            gen_dict=gen_dict,
            prompt_base=prompt,
            dry_run=False,
        )

        results = []
        for d in tqdm(data):
            res = be.predict(**d, gen_dict=gen_dict)
            output = {**d, **res}
            write_jsonl_into_file([output], outfile, mode='a', do_tqdm=False)
            results.append(output)

        # if postprocess:
        #     be.postprocess(outfile)

    print('Done')


def explain():
    fnames = {
        'sbf': '/vol/bitbucket/aeg19/ag/datasets/data-aux/sbf/jsonl.cxpr.splits/test.jsonl',
        # 'unib': '/vol/bitbucket/aeg19/ag/datasets/data-aux/jigsaw-task2/jsonl.cxpr.splits/train.jsonl',
        # 'mhs': '/vol/bitbucket/aeg19/ag/datasets/data-aux/measuring_hate_speech/jsonl.cxpr.splits/test.jsonl',
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
    max_samples = 9
    demos_threshold = 0.8
    prompt_types = [1,2,3]
    exclude_files = [
        demos_file,
        '/data2/ag/home/ToxEx/data/generations/2022-06-07-10-36-32.jsonl',
    ]

    data = load_data(fnames, fields, max_samples, exclude_files, require_target_group=True)

    for prompt_type in prompt_types:
        be = Explainer(
            demos_file=demos_file,
            api_key=api_key,
            target_groups=target_groups,
            target_group_map_file=target_group_map_file,
            prompt_base=prompt_base,
            demos_threshold=demos_threshold,
            prompt_type=prompt_type,
            dry_run=True,
        )

        results = []
        for d in tqdm(data):
            res = be.predict(**d, gen_dict=gen_dict)
            output = {**d, **res}
            write_jsonl_into_file([output], outfile, mode='a', do_tqdm=False)
            results.append(output)

    print('Done')


if __name__ == '__main__':
    binary()
    # explain()
    # postprocess('data/binary/2022-06-09-20-27-59.jsonl')