import pandas as pd
from utils import write_jsonl_into_file

def split_before_newlines(x):
    x = x.strip()
    if x.startswith('Th'):
        return x
    elif '\n\n' in x:
        return x.split('\n\n')[1]
    else:
        return x

path = '/data2/ag/home/ToxEx/data/generations/2022-06-07-10-36-32.jsonl'
df = pd.read_json(path, lines=True)
df[~df['text_response'].apply(lambda x: x.strip().startswith('Th'))]['text_response'].tolist()

df['explanation'] = df['explanation'].apply(split_before_newlines)
df['text_response'] = df['text_response'].apply(split_before_newlines)
write_jsonl_into_file(df.to_dict(orient='records'), path.replace('.jsonl', '-proc.jsonl'))
