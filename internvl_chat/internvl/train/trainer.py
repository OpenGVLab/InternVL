import transformers
from transformers import Trainer
from .utils import profiler
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

class CustomTrainer(Trainer):
    def __init__(self, use_cuda_graph = False, **kwargs):
        super().__init__(**kwargs)
        self.profiler = profiler(enable_profile=True, profile_dir='tracing', global_step=0)
        self.use_cuda_graph = use_cuda_graph
        self.graph_to_build = True
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.graph_to_build and self.use_cuda_graph:
            from cuda_graph_utils import pack_model4cuda_graph
            pack_model4cuda_graph(model=model, batch_size=self.args.train_batch_size, seq_len=self.tokenizer.model_max_length)
            self.graph_to_build = False
        if self.profiler.step_num == 0:
            self.profiler.start()
        res = super().training_step(model, inputs)
        self.profiler.step()
        if self.profiler.step_num == 10:
            self.profiler.stop()
        return res