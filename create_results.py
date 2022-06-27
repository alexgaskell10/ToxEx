from unittest import result
import numpy as np
import pandas as pd

from toxex.mturk.analyze_responses import compute_krip_alpha
from toxex.mturk.analyze_turkers import (
    compute_fleiss_kappa, 
    compute_turker_agreement,
    find_bad_annots, 
    gold_turker_analysis,
    compute_turker_gold_agreement,
    cheat_analysis,
)
from toxex.mturk.common import (
    load_data,
)
from toxex.mturk.analyze_pre_entry import (
    compute_pre_entry_performance_by_acceptance,
)
from toxex.mturk.analyze_responses import (
    sample_level_performance,
    missing_annotations
)
from toxex.analyze_binary import (
    binary_significance_tests,
)


def inter_annotator_table():
    results = {}
    # Compute unfiltered metrics
    _, _, _, _, df_answers, df_answers_unag = load_data([1], filter_workers=False)
    results.update({'unfiltered_'+k:v for k,v in compute_fleiss_kappa(df_answers).items()})
    # results.update({'unfiltered_'+k:v for k,v in compute_krip_alpha(df_answers_unag).items()})
    # # results.update({'unfiltered_'+k:v for k,v in compute_turker_agreement(df_answers).items()})
    # # Compute filtered metrics
    _, _, _, _, df_answers, df_answers_unag = load_data([1,2,3], filter_workers=True)
    results.update({'filtered_'+k:v for k,v in compute_fleiss_kappa(df_answers).items()})
    # results.update({'filtered_'+k:v for k,v in compute_krip_alpha(df_answers_unag).items()})
    # # results.update({'filtered_'+k:v for k,v in compute_turker_agreement(df_answers).items()})
    print(results)

    # results = {'unfiltered_fk_5-level': 0.10480322594477764, 'unfiltered_fk_3-level': 0.18456027458696825, 'unfiltered_fk_2-level': 0.20488591354345492, 'unfiltered_ka_nominal_metric_2-level': 0.08527605346930445, 'unfiltered_ka_nominal_metric_3-level': 0.07682040181805494, 'unfiltered_ka_nominal_metric_5-level': 0.03823686238747015, 'unfiltered_ka_interval_metric_2-level': 0.08527605346930445, 'unfiltered_ka_interval_metric_3-level': 0.0946370015275293, 'unfiltered_ka_interval_metric_5-level': 0.11180153675482096, 'unfiltered_agree-3_2-level': 0.46412213740458014, 'unfiltered_agree-2_2-level': 0.5358778625954198, 'unfiltered_agree-0_2-level': 0.0, 'unfiltered_agree-3_3-level': 0.4396946564885496, 'unfiltered_agree-2_3-level': 0.5043256997455471, 'unfiltered_agree-0_3-level': 0.05597964376590331, 'unfiltered_agree-3_5-level': 0.1811704834605598, 'unfiltered_agree-2_5-level': 0.5389312977099237, 'unfiltered_agree-0_5-level': 0.27989821882951654, 'filtered_fk_5-level': 0.1542235117182419, 'filtered_fk_3-level': 0.2727722292632634, 'filtered_fk_2-level': 0.29185260665995294, 'filtered_ka_nominal_metric_2-level': 0.2921097407897061, 'filtered_ka_nominal_metric_3-level': 0.2730362916346273, 'filtered_ka_nominal_metric_5-level': 0.15453062010178642, 'filtered_ka_interval_metric_2-level': 0.2921097407897061, 'filtered_ka_interval_metric_3-level': 0.3243367424797825, 'filtered_ka_interval_metric_5-level': 0.3817839522804115, 'filtered_agree-3_2-level': 0.5305010893246187, 'filtered_agree-2_2-level': 0.46949891067538124, 'filtered_agree-0_2-level': 0.0, 'filtered_agree-3_3-level': 0.5098039215686274, 'filtered_agree-2_3-level': 0.4433551198257081, 'filtered_agree-0_3-level': 0.046840958605664486, 'filtered_agree-3_5-level': 0.21459694989106753, 'filtered_agree-2_5-level': 0.5348583877995643, 'filtered_agree-0_5-level': 0.25054466230936817}
    results = {'unfiltered_fk_5-level': 0.10642463994361782, 'unfiltered_fk_3-level': 0.18616610608354522, 'unfiltered_fk_2-level': 0.20640258720173565, 'unfiltered_ka_nominal_metric_2-level': 0.08527605346930445, 'unfiltered_ka_nominal_metric_3-level': 0.07682040181805494, 'unfiltered_ka_nominal_metric_5-level': 0.03823686238747015, 'unfiltered_ka_interval_metric_2-level': 0.08527605346930445, 'unfiltered_ka_interval_metric_3-level': 0.0946370015275293, 'unfiltered_ka_interval_metric_5-level': 0.11180153675482096, 'filtered_fk_5-level': 0.17434805122868113, 'filtered_fk_3-level': 0.32874232890643057, 'filtered_fk_2-level': 0.36275683395888003, 'filtered_ka_nominal_metric_2-level': 0.306327816863987, 'filtered_ka_nominal_metric_3-level': 0.28060723849146574, 'filtered_ka_nominal_metric_5-level': 0.15709028978145778, 'filtered_ka_interval_metric_2-level': 0.306327816863987, 'filtered_ka_interval_metric_3-level': 0.333044422353706, 'filtered_ka_interval_metric_5-level': 0.38426339082439287}
    rows = [('filtered',2), ('unfiltered',2), ('filtered',3), ('unfiltered',3), ('filtered',5), ('unfiltered',5)]
    # cols = ['fk','ka_nominal','ka_interval','agree-3','agree-2','agree-0']
    cols = ['fk','ka_nominal','ka_interval']
    data = []
    for fil, r in rows:
        row = []
        for c in cols:
            for k,v in results.items():
                if f'{r}-level' in k and k.startswith(fil) and f'{c}_' in k:
                    row.append(v)
        data.append(row)

    df = pd.DataFrame(data)
    df = df.round(3) * 100
    # df.index = rows
    df.index = pd.MultiIndex.from_tuples(rows, names=['filtered','group_level'])
    df.columns = cols
    df = df.sort_index(level=0, axis=0)
    print(df.to_latex(index=True))


def gold_turker_performance(df_answers_unag):
    results = {}
    results.update(compute_turker_gold_agreement(df_answers_unag))
    results.update(compute_pre_entry_performance_by_acceptance())
    
    # results = {'main_passed-2': 0.7403189066059226, 'main_failed-2': 0.5119305856832972, 'main_passed-3': 0.7061503416856492, 
    #         'main_failed-3': 0.4837310195227766, 'preentry_passed-2': 0.8045918367346939, 'preentry_failed-2': 0.5854330708661417, 
    #         'preentry_passed-3': 0.7688775510204081, 'preentry_failed-3': 0.5409448818897638}
    
    cols = ['preentry_all','preentry_passed','preentry_failed','main_all','main_passed','main_failed']
    rows = [2,3]
    data = []
    for r in rows:
        row = []
        for c in cols:
            for k,v in results.items():
                if c in k and str(r) in k:
                    row.append(v)
        data.append(row)

    df = pd.DataFrame(data)
    df = df.round(3) * 100
    df['scale'] = rows
    df.columns = cols + ['scale']
    df = df[['scale']+cols]
    print(df.to_latex(index=False))


def gpt3_explanation_results(df_answers):
    # results_agree = sample_level_performance(df_answers.copy(), agg_level=2)
    # results_agree.columns = pd.MultiIndex.from_tuples(
    #     [(c,'AR (%)') for c in results_agree.columns])

    # results_strongagree = sample_level_performance(df_answers.copy(), agg_level=6)
    # results_strongagree.columns = pd.MultiIndex.from_tuples(
    #     [(c,'SAR (%)') for c in results_strongagree.columns])

    # results = pd.concat((results_agree, results_strongagree), axis=1)
    # results.sort_index(axis=1, ascending=False, inplace=True)
    
    results = sample_level_performance(df_answers.copy(), agg_level=2)

    print((results * 100).round(1).to_latex())
    
    print('Done')


def binary_table_results():
    gpt3_path = '/data2/ag/home/ToxEx/data/binary/aaa-2022-06-10-09-51-35-all/cx-output.json'
    df_gpt3 = pd.read_json(gpt3_path).sort_values('name')
    df_gpt3['system'] = 'gpt3'
    toxi_path = '/data2/ag/home/ag/experiments/mhs_unib_sbf_toxicity/unintended_bias_measuring_hate_speech_sbf/run4/cx-output.json'
    df_toxi = pd.read_json(toxi_path).sort_values('name')
    df_toxi['name'] = df_toxi['name'].apply(lambda x: 'unib' if 'unintended_bias' in x 
        else 'mhs' if 'measuring' in x else 'sbf')
    df_toxi['system'] = 'toxi'
    cols = ['name','system','accuracy','f1_macro_avg']
    df = pd.concat((df_gpt3[cols], df_toxi[cols]))

    cols = ['mhs','sbf','unib']
    rows = [0,1,2]
    df['dset'] = df['name'].apply(lambda x: 
        x.replace('.jsonl','').split('-')[0] if any(c in x for c in cols) else 'all')
    df['pid'] = df['name'].apply(lambda x: 
        x.replace('.jsonl','').split('-')[1] if any(str(r) in x and '-' in x for r in rows) else 
        x.replace('.jsonl','') if 'pid_' in x else 'all')
    
    res = df[['accuracy','f1_macro_avg','dset','pid','system']].pivot(
        columns='dset', index=['system','pid'], values=['accuracy','f1_macro_avg'])
    res.columns.names = ['metric','dset']
    res = res.reorder_levels([1,0], axis=1)
    res = res.sort_index(axis=1, ascending=False)
    res = res.iloc[[1,2,3,0,4], [1,0,3,2,5,4]]

    res = res.round(3) * 100
    print(res.to_latex(index=True))

    binary_significance_tests()


if __name__ == '__main__':

    df_samples, df_output, df_hits, df_hits_unag, df_answers, df_answers_unag = load_data(
        [1,2,3], filter_workers=True)

    # inter_annotator_table()
    # gold_turker_analysis(df_answers_unag)
    # gold_turker_performance(df_answers_unag)
    gpt3_explanation_results(df_answers)
    # binary_table_results()    
    # missing_annotations(df_answers_unag)
    # cheat_analysis(df_answers_unag)
    # find_bad_annots(df_answers_unag, df_answers)
