import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class TwoMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit_hidden_size = 3200
        self.mlp1 = nn.Sequential(
            nn.Linear(self.vit_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, inputs):
        images, queries = inputs
        images = self.mlp1(images)
        queries = self.mlp2(queries)
        out = torch.cat([queries, images], dim=1)
        assert out.size(1) == 576 + 96, f"Expected 576+96, got {out.size(1)}"

        return out


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu*', projector_type)
    use_ln = "ln" in projector_type
    print("use LN for projection: ", use_ln)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = []
        if use_ln:
            modules.append(nn.LayerNorm(config.mm_hidden_size))
        modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    if projector_type == 'two_mlp':
        return TwoMLP(config)

    raise ValueError(f'Unknown projector type: {projector_type}')
