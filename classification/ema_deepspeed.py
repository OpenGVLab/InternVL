import torch
import torch.nn as nn
import deepspeed
from deepspeed.runtime.zero import GatheredParameters
from contextlib import contextmanager


class EMADeepspeed(nn.Module):
    """ migrated from https://github.com/microsoft/DeepSpeed/issues/2056
    """

    def __init__(self, model, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.decay = decay
        self.num_updates = 0 if use_num_updates else -1

        with GatheredParameters(model.parameters(), fwd_module=self):
            for name, p in model.named_parameters():
                if p.requires_grad:
                    # remove as '.'-character is not allowed in buffers
                    s_name = name.replace('.', '')
                    self.m_name2s_name.update({name: s_name})
                    self.register_buffer(s_name, p.clone().detach().data)
                    # remove as '.'-character is not allowed in buffers
        self.collected_params = []

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay
        shadow_params = dict(self.named_buffers())

        with torch.no_grad():
            with GatheredParameters(model.parameters()):
                if deepspeed.comm.get_rank() == 0:
                    m_param = dict(model.named_parameters())

                    for key in m_param:
                        if m_param[key].requires_grad:
                            sname = self.m_name2s_name[key]
                            shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                            shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                        else:
                            assert not key in self.m_name2s_name

    def copy_to(self, model):
        shadow_params = dict(self.named_buffers())
        with GatheredParameters(model.parameters(), modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                m_param = dict(model.named_parameters())
                for key in m_param:
                    if m_param[key].requires_grad:
                        m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
                    else:
                        assert not key in self.m_name2s_name

    def store(self, model):
        """
        Save the current parameters for restoring later.
        Args:
          model: A model that parameters will be stored
        """
        with GatheredParameters(model.parameters()):
            if deepspeed.comm.get_rank() == 0:
                parameters = model.parameters()
                self.collected_params = [param.clone() for param in parameters]

    def restore(self, model):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          model: A model that to restore its parameters.
        """
        with GatheredParameters(model.parameters(), modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                parameters = model.parameters()
                for c_param, param in zip(self.collected_params, parameters):
                    param.data.copy_(c_param.data)

    @contextmanager
    def activate(self, model):
        try:
            self.store(model)
            self.copy_to(model)
            yield
        finally:
            self.restore(model)
