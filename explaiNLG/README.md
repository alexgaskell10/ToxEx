# explaiNLG

Code to train a student model for producing NLG explanations for toxicity. This code is adapted from the Huggingface `examples` - for more information you can see the relevant directories within the [Huggingface Github](https://github.com/huggingface/transformers/tree/main/examples/pytorch).

Key details:
- GPT2 / GPTNeo / GPTJ: `language-modelling`
    - Training the model with demos: `explaiNLG/language-modeling/lm_train_with_demos.sh`
    - Training the model without demos: `explaiNLG/language-modeling/lm_train_wo_demos.sh`
    - Obtain predictions from a trained model: `explaiNLG/language-modeling/lm_predict.sh`
- T5: `summarization`
    - Training the model with demos: `explaiNLG/summarization/sum_train_with_demos.sh`
    - Training the model without demos: `explaiNLG/summarization/sum_train_wo_demos.sh`
    - Obtain predictions from a trained model: `explaiNLG/summarization/sum_predict.sh`
- Evaluate a set of generated predictions using `explaiNLG/eval`

Data for training the model is currently found in `/data2/ag/home/ag/datasets/data-aux/gpt3_explanations` (not currently integrated into the `datasets` repo).

### Setup

- Create environment
- `pip install -r requirements.txt`