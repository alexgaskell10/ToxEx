import json, os, sys
import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
import random
import jsonlines
import tqdm


np.random.seed(1)
random.seed(1)


def write_jsonl_into_file(data: List[dict], fname: str) -> None:
    """Writes a list of dictionaries into the given file path as .jsonl."""
    with jsonlines.open(str(fname), mode="w") as f:
        for line in tqdm.tqdm(data, unit="line"):
            f.write(line)


datasets = ['sbf', 'jigsaw-task2', 'measuring_hate_speech']
data_base_path = '/data2/ag/home/ag/datasets/data/XXX/jsonl.cxpr.splits/test.jsonl'
n_samples = 512
outdir_base = Path('data/test_samples')
outdir_base.mkdir(exist_ok=True)

# dataset = datasets[0]
for dataset in datasets:

    data_path = Path(data_base_path.replace('XXX', dataset))
    df = pd.read_json(data_path, orient='records', lines=True)
    df_samples = df.sample(512)

    print(dataset)
    print('Toxicity:\n', df_samples["toxicity"].value_counts())
    print('Target group:\n', df_samples["target_group"].apply(sum).value_counts())

    outfile = outdir_base / dataset / f'test_sample_{len(df_samples)}.jsonl'
    outfile.parent.mkdir(exist_ok=True)
    write_jsonl_into_file(df_samples.to_dict(orient="records"), outfile)
