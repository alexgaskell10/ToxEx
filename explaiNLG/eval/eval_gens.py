from optparse import Option
import os, sys, json
import numpy as np
import pandas as pd
import nltk
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict
from dataclasses import dataclass, field
from tqdm import tqdm

from datasets import load_dataset, load_metric, Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel
from transformers.hf_argparser import HfArgumentParser


@dataclass
class Arguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    tokenizer_name: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    generations_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    references_file: Optional[str] = field(default=None)
    batch_size: Optional[int] = field(default=8)
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    outfile: Optional[str] = field(default=None)
    write_all: Optional[bool] = field(default=False)


def main():

    # sys.argv = [
    #     "/vol/bitbucket/aeg19/ag/transformers/examples/pytorch/summarization/eval_gens.py",
    #     "--generations_file", "/data2/ag/home/ToxEx/data/auto_eval/cands.txt",
    #     "--references_file", "/data2/ag/home/ToxEx/data/auto_eval/refs.csv",
    #     "--batch_size", "1",
    #     "--write_all",
    #     "--summary_column", "explanation_y",
    # ]

    parser = HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()

    # Load dataset
    df_gens = pd.read_csv(args.generations_file, sep='-:-:', header=None, skip_blank_lines=False)
    df_refs = pd.read_csv(args.references_file, skip_blank_lines=False)
    df_gens.columns = ['text']

    assert len(df_gens) == len(df_refs)
    if 'text' in df_refs.columns:
        print('Column named "text" found in references df. Droppping this column.' )
        df_refs.drop('text', inplace=True, axis=1)
    df = pd.concat((df_refs, df_gens), axis=1)
    df['text'].fillna('', inplace=True)
    
    dataset = Dataset.from_pandas(df)
    if args.summary_column:
        response_cols = [args.summary_column]
    else:
        response_cols = [k for k in dataset.features.keys() if k.startswith('response')]

    rouge = load_metric("rouge")
    bertscore = load_metric("bertscore")

    def rouge_postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_rouge(eval_preds):
        preds, labels = eval_preds

        # Some simple post-processing
        decoded_preds, decoded_labels = rouge_postprocess_text(preds, labels)

        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=False
        )
        # Extract a few results from ROUGE
        result = {k: [x.fmeasure for x in v] for k,v in result.items()}
        return result
        
    def compute_bertscore(eval_preds):
        preds, labels = eval_preds

        result = bertscore.compute(predictions=preds, references=labels, lang='en')
        result = {"bertscore_f1": result["f1"]}
        return result

    def aggregate(scores, results, aggr_method):
        for k in scores[0].keys():
            agg = [aggr_method(*xs) for xs in zip([rr[k] for rr in scores])]
            results[k] = np.mean(agg)
            results['eval_samples'] = len(agg)
        return results

    def compute_metrics(eval_preds, aggr_method=max):
        preds, all_labels = eval_preds
        if not isinstance(all_labels, tuple):
            all_labels = (all_labels,)
        results = {}

        # Compute rouge scores
        rouge_results = []
        for labels in all_labels:
            rouge_results.append(compute_rouge((preds, labels)))

        # Aggregate scores
        results = aggregate(rouge_results, results, aggr_method)

        # Compute bertscore scores
        bertscore_results = []
        for labels in all_labels:
            bertscore_results.append(compute_bertscore((preds, labels)))

        # Aggregate scores
        results = aggregate(bertscore_results, results, aggr_method)

        return results

    if args.write_all: 
        bsz = 1
    else:
        bsz = 48

    data = DataLoader(dataset, batch_size=bsz)


    results = []
    for n, batch in enumerate(tqdm(data)):
        preds = [p if p else '' for p in batch['text']]
        refs = tuple(batch[k] for k in response_cols)
        batch_results = compute_metrics((preds, refs), max)
        results.append(batch_results)
        # if n == 3:
        #     break

    # Aggregate results
    output = {}
    for k in results[0].keys():
        if k == 'eval_samples':
            output[k] = sum([r[k] for r in results])
        else:
            output[k] = np.mean([r[k] for r in results])
    
    if args.write_all:
        output["all_scores"] = results

    # Write to file
    if args.outfile:
        outfile = args.outfile
    else:
        outfile = args.generations_file.replace("generated_predictions.txt", "augmented_eval.json")
    print('Writing output to ', outfile)
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=4)

    print('Done')


if __name__ == '__main__':
    main()
