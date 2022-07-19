import torch
from transformers.data.data_collator import DataCollatorForSeq2Seq


class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, *args, **kwargs):
        features = super().__call__(features, *args, **kwargs)
        if 'labels_attention_mask' in features:         # Pad labels attention mask so it is same shape as labels
            lam = features['labels_attention_mask']
            labels = features['labels']
            if lam.size(1) < labels.size(1):
                features['labels_attention_mask'] = torch.cat((
                    lam, torch.zeros_like(labels)[:,:labels.size(1) - lam.size(1)]
                ), dim=1)
        return features
