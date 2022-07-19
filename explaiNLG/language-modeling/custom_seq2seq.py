from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.deepspeed import is_deepspeed_zero3_enabled

from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    ''' Overrides Seq2SeqTrainer to allow conditional language model
        training with label smoothing.
    '''
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            pad_id = self.tokenizer.pad_token_id
            sep_id = self.tokenizer.sep_token_id
            bos_id = self.tokenizer.bos_token_id
            # Input should look like: [input_ids [SEP] [PAD] ... [PAD] [BOS] label_ids [EOS] [PAD] ... [PAD]]
            # Labels should look like: [label_ids [EOS] -100 ... -100]
            input_ids = inputs.pop("input_ids")
            labels = inputs.pop("labels")
            bos_ids = torch.full_like(labels, bos_id)[:,:1]
            labels = torch.cat([bos_ids, labels], dim=1)
            gen_ids = torch.where(labels == -100, pad_id, labels)
            inputs["input_ids"] = torch.cat([input_ids, gen_ids], dim=1)

            # Attention mask should look like [[1] ... [1] [1] labels_attn_mask]
            attention_mask = inputs.pop("attention_mask")
            labels_attention_mask = inputs.pop("labels_attention_mask")
            bos_mask = torch.full_like(attention_mask, 1)[:,:1]
            inputs["attention_mask"] = torch.cat([attention_mask.fill_(1), bos_mask, labels_attention_mask], dim=1)

        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:

            if isinstance(model, GPT2LMHeadModel) or isinstance(model, GPTNeoForCausalLM) \
                or isinstance(model, GPTJForCausalLM):
                # Only train the generation, not the prompt
                outputs.logits = outputs.logits[..., input_ids.size(1):, :]
                # Shift so that tokens < n predict n
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                outputs.logits = shift_logits
                labels = shift_labels
                # target_len = max(shift_logits.size(1), shift_labels.size(1))
                # outputs.logits = self._pad_tensors_to_max_len(shift_logits, target_len, 0)
                # labels = self._pad_tensors_to_max_len(shift_labels, target_len, -100)

                # TODO: fix this so it only does this for text response, not the prompt

            loss = self.label_smoother(outputs, labels)
            outputs.loss = loss
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": (self._max_length if self._max_length is not None else self.model.config.max_length) * 2,     # FIXME: have different lengths for source and target rather than just multiplying by 2
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # if "attention_mask" in inputs:
        #     gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        # Generation inputs should look like [input_ids [SEP] [PAD] .. [PAD] [BOS]]
        bos_id = self.tokenizer.bos_token_id
        bos_ids = torch.full_like(generation_inputs, bos_id)[:,:1]
        generation_inputs = torch.cat([generation_inputs, bos_ids], dim=1)

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # Strip the input tokens from the generated tokens
        generated_tokens = generated_tokens[:,generation_inputs.size(1):]

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        # with torch.no_grad():
        #     with self.autocast_smart_context_manager():
        #         outputs = model(**inputs)
        #     if has_labels:
        #         if self.label_smoother is not None:
        #             loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
        #         else:
        #             loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
        #     else:
        #         loss = None
        loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            # Labels should look like: [label_ids [EOS] -100 ... -100] 
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length, pad_token_id=None):
        if not pad_token_id:
            if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
                # If PAD token is not defined at least EOS token has to be defined
                pad_token_id = (
                    self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                )
            else:
                if self.model.config.pad_token_id is not None:
                    pad_token_id = self.model.config.pad_token_id
                else:
                    raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        if len(tensor.shape) == 2:
            padded_tensor = pad_token_id * torch.ones(
                (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:, : tensor.shape[-1]] = tensor
        elif len(tensor.shape) == 3:
            padded_tensor = pad_token_id * torch.ones(
                (tensor.shape[0], max_length, tensor.shape[2]), dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:, : tensor.shape[1], :] = tensor
            
        return padded_tensor
