import pandas as pd
from scipy.stats import ttest_ind


def load_data():
    eval_path = '/data2/ag/home/ToxEx/data/binary/aaa-2022-06-10-09-51-35-all/cx-output.json'
    data = pd.read_json(eval_path).sort_values('name')
    return data

def binary_significance_tests():
    fnames = [
        (
            'mhs',
            '/data2/ag/home/ag/experiments/mhs_unib_sbf_toxicity/unintended_bias_measuring_hate_speech_sbf/run4/best_sbf_model.task+toxicity-ds+measuring-test.jsonl',
            '/data2/ag/home/ToxEx/data/binary/aaa-2022-06-10-09-51-35-all/mhs-pid_1.jsonl',
        ),
        (   
            'sbf',
            '/data2/ag/home/ag/experiments/mhs_unib_sbf_toxicity/unintended_bias_measuring_hate_speech_sbf/run4/best_sbf_model.task+toxicity-ds+sbf-test.jsonl',
            '/data2/ag/home/ToxEx/data/binary/aaa-2022-06-10-09-51-35-all/sbf-pid_1.jsonl',
        ),
        (   
            'unib',
            '/data2/ag/home/ag/experiments/mhs_unib_sbf_toxicity/unintended_bias_measuring_hate_speech_sbf/run4/best_sbf_model.task+toxicity-ds+unintended_bias-test.jsonl',
            '/data2/ag/home/ToxEx/data/binary/aaa-2022-06-10-09-51-35-all/unib-pid_1.jsonl',
        ),
    ]
    results = {}
    for name, fname1, fname2 in fnames:
        df1 = pd.read_json(fname1, orient='records', lines=True)
        df2 = pd.read_json(fname2, orient='records', lines=True)
        
        df1['label'] = df1['input'].apply(lambda x: x['toxicity'])

        correct1 = (df1['toxicity_pred_label'] == df1['label']).astype(int)
        correct2 = (df2['pred_label_id'] == df2['bin_label_id']).astype(int)

        ttest = ttest_ind(correct1, correct2)
        results[name] = ttest

    return results