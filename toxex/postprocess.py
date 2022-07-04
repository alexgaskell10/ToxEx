import re
import pandas as pd
from pathlib import Path
from .utils import write_jsonl_into_file

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
    df['uid'] = df['id'] + '::' + df['prompt_id'].astype(str)

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
    input_cols = ['uid', 'text', 'prompt_base']
    df_cxpy = df[['dataset','prompt_id']+list(col_map)].rename(columns=col_map)
    df_cxpy['input'] = df[input_cols].to_dict(orient='records')

    # Write outputs
    outdir = Path(str(file).replace('.jsonl',''))
    print('Writing outputs to', outdir)
    outdir.mkdir(exist_ok=True)
    write_jsonl_into_file(df_cxpy.to_dict(orient='records'), Path(*file.parts[:2],file.stem,'all.jsonl'))
    
    for dset in df_cxpy['dataset'].drop_duplicates().values:
        outpath = outdir / f'{dset}.jsonl'
        rows = df_cxpy[(df_cxpy['dataset'] == dset)]
        write_jsonl_into_file(rows.to_dict(orient='records'), outpath)
    
    for pid in df_cxpy['prompt_id'].drop_duplicates().values:
        outpath = outdir / f'pid_{pid}.jsonl'
        rows = df_cxpy[(df_cxpy['prompt_id'] == pid)]
        write_jsonl_into_file(rows.to_dict(orient='records'), outpath)
    
    for dset,pid in df_cxpy[['dataset', 'prompt_id']].drop_duplicates().values:
        outpath = outdir / f'{dset}-pid_{pid}.jsonl'
        rows = df_cxpy[(df_cxpy['dataset'] == dset) & (df_cxpy['prompt_id'] == pid)]
        write_jsonl_into_file(rows.to_dict(orient='records'), outpath)

    print("done")

