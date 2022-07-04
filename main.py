import os
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

from toxex.gpt3.binary import BinaryPredictor
from toxex.gpt3.explain import Explainer
from toxex.postprocess import postprocess
from toxex.utils import (
    set_seed,
    write_jsonl_into_file,
    dt_now,
    load_data,
    read_txt_lines,
)


set_seed(1, 1)

api_key = os.getenv("OPENAI_API_KEY")


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
    exclude_files = [
        '/data2/ag/home/ToxEx/data/binary/aaa-2022-06-09-22-12-58.jsonl',
        '/data2/ag/home/ToxEx/data/binary/aaa-2022-06-10-09-51-35.jsonl',
    ]

    data = load_data(fnames, fields, max_samples, exclude_files=exclude_files)

    for prompt in prompts:
        be = BinaryPredictor(
            api_key=api_key,
            gen_dict=gen_dict,
            prompt_base=prompt,
            dry_run=True,
        )

        results = []
        for d in tqdm(data):
            res = be.predict(**d, gen_dict=gen_dict)
            output = {**d, **res}
            write_jsonl_into_file([output], outfile, mode='a', do_tqdm=False)
            results.append(output)

    print('Done')


def explain():
    fnames = {
        'sbf': '/data2/ag/home/ag/datasets/data-aux/sbf/jsonl.cxpr.splits/test.jsonl',
        # 'unib': '/data2/ag/home/ag/datasets/data-aux/jigsaw-task2/jsonl.cxpr.splits/train.jsonl',
        # 'mhs': '/data2/ag/home/ag/datasets/data-aux/measuring_hate_speech/jsonl.cxpr.splits/test.jsonl',
    }
    outfile = Path(f'data/generations/{dt_now()}.jsonl')
    outfile.parent.mkdir(exist_ok=True)
    print('Output will be written to:', outfile)

    args = argparse.Namespace(**yaml.load(open('configs/explainer.yml'), Loader=yaml.SafeLoader))
    tgt_groups_names = read_txt_lines('resources/target_group_names.txt')

    fields = ['id', 'text', 'toxicity', 'target_group']
    max_samples = 5
    exclude_files = [
        args.demos_fname,
        # '/data2/ag/home/ToxEx/data/generations/2022-06-07-10-36-32.jsonl',
    ]
    data = load_data(fnames, fields, max_samples, exclude_files, require_target_group=True)

    # prompt_types = [1,2,3]
    prompt_types = [3]
    for prompt_type in prompt_types:
        be = Explainer(**vars(args), prompt_type=prompt_type, dry_run=True)

        results = []
        for d in tqdm(data):
            res = be.predict(
                id = d['id'],
                text = d['text'],
                toxicity = d['toxicity'],
                tgt_groups_names = tgt_groups_names,
                tgt_groups_scores = d['target_group'],
            )
            output = {**d, **res}
            write_jsonl_into_file([output], outfile, mode='a', do_tqdm=False)
            results.append(output)

    print('Done')


if __name__ == '__main__':
    # binary()
    explain()
    # postprocess('data/binary/aaa-2022-06-10-09-51-35-all.jsonl')