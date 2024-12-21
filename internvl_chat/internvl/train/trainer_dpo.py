# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple, Union

import deepspeed
import torch
from torch import nn
from torch.utils.data import ConcatDataset
from trl import DPOTrainer
from trl.trainer.utils import RunningMoments, pad_to_length


def _map(self, *args, **kwargs):
    return self


ConcatDataset.map = _map


class MultimodalDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.loss_type != 'bco_pair' and 'bco_pair' in self.loss_type:
            self.running = RunningMoments(self.accelerator)

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch['chosen_labels'].shape[1], batch['rejected_labels'].shape[1])
        else:
            max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])

        for k in batch:
            if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
                if 'labels' in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith('_input_ids'):
                    pad_value = padding_value
                elif k.endswith('_attention_mask'):
                    pad_value = 0
                concatenated_key = k.replace('chosen', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
                if 'labels' in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith('_input_ids'):
                    pad_value = padding_value
                elif k.endswith('_attention_mask'):
                    pad_value = 0
                concatenated_key = k.replace('rejected', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch['concatenated_input_ids'] = batch['prompt_input_ids'].repeat(2, 1).to(device=device)
            concatenated_batch['concatenated_attention_mask'] = (
                batch['prompt_attention_mask'].repeat(2, 1).to(device=device)
            )

        if 'pixel_values' in batch:
            concatenated_batch['pixel_values'] = batch['pixel_values'].repeat(2, 1, 1, 1)
            concatenated_batch['image_flags'] = batch['image_flags'].repeat(2)

        return concatenated_batch

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch['chosen_labels'].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs['labels'] = concatenated_batch['concatenated_labels']
            model_kwargs['decoder_input_ids'] = concatenated_batch.pop('concatenated_decoder_input_ids', None)

        if self.is_vision_model:
            model_kwargs['pixel_values'] = concatenated_batch['pixel_values']
            model_kwargs['pixel_attention_mask'] = concatenated_batch['pixel_attention_mask']

        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True

        outputs = model(
            input_ids=concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask'],
            pixel_values=concatenated_batch['pixel_values'],
            image_flags=concatenated_batch['image_flags'],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch['concatenated_labels'],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch['concatenated_labels'].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        if self.loss_type == 'ipo':
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    def _prepare_deepspeed_orig(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs['zero_optimization']['stage'] != 3:
            config_kwargs['zero_optimization']['stage'] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if config_kwargs['zero_optimization']['stage'] == 3:
            print('Enable DPOTrainer._prepare_deepspeed')
            return self._prepare_deepspeed_orig(model)

        print('Disable DPOTrainer._prepare_deepspeed')
        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        model = model.to(self.accelerator.device)
        return model

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal['train', 'eval'] = 'train',
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            'reference_chosen_logps' in batch
            and 'reference_rejected_logps' in batch
            and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch['reference_chosen_logps']
            reference_rejected_logps = batch['reference_rejected_logps']
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        if ',' in self.loss_type:
            loss_type = self.loss_type
            loss_type_list = loss_type.split(',')

            losses, chosen_rewards, rejected_rewards = 0, 0, 0
            for curr_type in loss_type_list:
                self.loss_type = curr_type
                curr_losses, curr_chosen_rewards, curr_rejected_rewards = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                )
                curr_weight = getattr(self.args, f'{curr_type}_loss_weight')
                losses = losses + curr_losses * curr_weight
                chosen_rewards = chosen_rewards + curr_chosen_rewards * curr_weight
                rejected_rewards = rejected_rewards + curr_rejected_rewards * curr_weight

            self.loss_type = loss_type
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            # losses = losses * self.args.rpo_alpha + policy_nll_loss
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        prefix = 'eval_' if train_eval == 'eval' else ''
        metrics[f'{prefix}rewards/chosen'] = chosen_rewards.mean().cpu()
        metrics[f'{prefix}rewards/rejected'] = rejected_rewards.mean().cpu()
        metrics[f'{prefix}rewards/accuracies'] = reward_accuracies.mean().cpu()
        metrics[f'{prefix}rewards/margins'] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f'{prefix}logps/rejected'] = policy_rejected_logps.detach().mean().cpu()
        metrics[f'{prefix}logps/chosen'] = policy_chosen_logps.detach().mean().cpu()
        metrics[f'{prefix}logits/rejected'] = policy_rejected_logits.detach().mean().cpu()
        metrics[f'{prefix}logits/chosen'] = policy_chosen_logits.detach().mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f'{prefix}nll_loss'] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, 'router_aux_loss_coef', 0.0) * aux_loss, metrics

        return losses.mean(), metrics
