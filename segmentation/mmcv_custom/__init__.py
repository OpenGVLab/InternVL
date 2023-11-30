import torch

from .layer_decay_optimizer_constructor import \
    CustomLayerDecayOptimizerConstructor

__all__ = ['CustomLayerDecayOptimizerConstructor',]


from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.optimizer.builder import OPTIMIZERS
from torch.distributed.optim import ZeroRedundancyOptimizer


@OPTIMIZERS.register_module()
class ZeroAdamW(ZeroRedundancyOptimizer):
    def __init__(self, params, optimizer_class=torch.optim.AdamW, **kwargs):
        super().__init__(params[0]['params'],
                         optimizer_class=optimizer_class,
                         parameters_as_bucket_view=True,
                         **kwargs)
        for i in range(1, len(params)):
            self.add_param_group(params[i])


@HOOKS.register_module()
class ZeroHook(Hook):
    def __init__(self, interval):
        self.interval = interval

    def after_epoch(self, runner):
        runner.optimizer.consolidate_state_dict(to=0)

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            runner.optimizer.consolidate_state_dict(to=0)


@HOOKS.register_module()
class ToBFloat16Hook(Hook):

    def before_run(self, runner):
        runner.model.module.backbone.to(torch.bfloat16)
        runner.model.module.decode_head.to(torch.float32)
        try:
            runner.model.module.auxiliary_head.to(torch.float32)
        except:
            pass
        print('hook:', runner.model.module.backbone.dtype)


@HOOKS.register_module()
class ToFloat16Hook(Hook):

    def before_run(self, runner):
        runner.model.module.backbone.to(torch.float16)
        runner.model.module.decode_head.to(torch.float32)
        try:
            runner.model.module.auxiliary_head.to(torch.float32)
        except:
            pass
        try:
            runner.model.module.neck.to(torch.float32)
        except:
            pass
        print('hook:', runner.model.module.backbone.dtype)
