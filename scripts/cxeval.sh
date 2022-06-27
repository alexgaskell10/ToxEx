dir=/data2/ag/home/ag/experiments/mhs_unib_sbf_toxicity/unintended_bias_measuring_hate_speech_sbf/run4
cxeval \
    --input_path $(find $dir -name *.jsonl) \
    --task toxicity > $dir/cx-output.json