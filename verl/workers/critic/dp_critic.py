# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic

remember that prediction lots should be shifted left by 1,
so the first prediction is conditional on the prompt, not prompt + first response token.
"""
import itertools
from typing import Iterable

import torch
import torch.distributed
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.critic import BasePPOCritic
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOCritic']

"""


TODO: support value bce loss, and one-step value model
value bce loss:
    + separate KL loss (as done in grpo). This is necessary because if reward is augmented with kl it will be outside the range of (0, 1)
    + new class 
one-step value model:
    + patch the attention mask to make the model only predict the value in the first head
"""


class DataParallelPPOCritic(BasePPOCritic):

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        print(f'Critic use_remove_padding={self.use_remove_padding}')

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                # NOTE: concatenate all sequences' non-pad input_ids in the batch.
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            **multi_modal_inputs,
                                            use_cache=False)  # prevent model thinks we are generating
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outpus_and_unpad(values_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                values = values[:, -response_length - 1:-1] # shift left by 1.
            else:
                output = self.critic_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            **multi_modal_inputs,
                                            use_cache=False)  # prevent model thinks we are generating
                values = output.logits
                values = values[:, -response_length - 1:-1].squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        self.critic_optimizer.step()
        return grad_norm

    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)
        responses = data.batch['responses']
        attention_mask = data.batch['attention_mask']
        response_length = responses.size(1)
        values = values * attention_mask[:, -response_length - 1:-1]

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]

        return values

    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu

                self.critic_optimizer.zero_grad()

                for data in micro_batches:
                    #Support all devices
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # critic device is cpu when using offload
                    input_ids = data['input_ids']
                    responses = data['responses']
                    attention_mask = data['attention_mask']
                    position_ids = data['position_ids']
                    values = data['values']
                    returns = data['returns']
                    response_length = responses.size(1)

                    eos_mask = attention_mask[:, -response_length - 1:-1] # shift left by 1

                    vpreds = self._forward_micro_batch(data)

                    # assert not torch.any(torch.isnan(vpreds)).item()

                    vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                         values=values,
                                                                         returns=returns,
                                                                         eos_mask=eos_mask,
                                                                         cliprange_value=self.config.cliprange_value)
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        # NOTE: this seems heuristic?
                        loss = vf_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = vf_loss / self.gradient_accumulation

                    loss.backward()

                    data = {
                        'critic/vf_loss': vf_loss.detach().item(),
                        'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                        'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
                    }

                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'critic/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.critic_optimizer.zero_grad()
        return metrics


class DataParallelPPOCriticOneStep(DataParallelPPOCritic):
    """
    predict a bernoulli value at the end of the prompt. (first token of the response)
    This should be considered a quick patch to the super class, for experimentation purposes.
    """
    def __init__(self, config, critic_module, critic_optimizer):
        super().__init__(config, critic_module, critic_optimizer)

        # dynamically change the compute_value_loss in update_critic
        def compute_sigmoid_loss(vpreds, returns, values, eos_mask, cliprange_value):
            """
            vpreds is expited
            """
            vf_losses = torch.nn.functional.binary_cross_entropy(vpreds, returns, reduction='none')
            vf_loss = torch.mean(vf_losses * eos_mask)
            return vf_loss, vf_loss

        core_algos.compute_value_loss = compute_sigmoid_loss

    def _forward_micro_batch(self, micro_batch):
        # responses = micro_batch['responses']
        # response_length = responses.size(-1)
        # input_ids = micro_batch['input_ids']
        # batch, seqlen = input_ids.shape
        # attention_mask = micro_batch['attention_mask']
        # position_ids = micro_batch['position_ids']
        
        # prompt_length = seqlen - response_length # should be 1024

        # # input_ids include the prompt and response and padding (left pad for prompt and right pad for response)
        # # attention mask is 0 for padding and 1 for non-padding

        # critic_attention_mask = attention_mask[:, :prompt_length + 1]
        # critic_responses = responses[:, 0:1]
        # critic_input_ids = input_ids[:, :prompt_length + 1]
        # critic_position_ids = position_ids[:, :prompt_length + 1]

        # micro_batch['responses'] = critic_responses
        # micro_batch['input_ids'] = critic_input_ids
        # micro_batch['attention_mask'] = critic_attention_mask
        # micro_batch['position_ids'] = critic_position_ids

        logits = super()._forward_micro_batch(micro_batch)

        # # undo
        # micro_batch['responses'] = responses
        # micro_batch['input_ids'] = input_ids
        # micro_batch['attention_mask'] = attention_mask
        # micro_batch['position_ids'] = position_ids

        return torch.sigmoid(logits)