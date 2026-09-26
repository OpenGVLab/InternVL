"""Small-config regressions for the vision encoder; no model download required."""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope='module', params=['internvl_chat', 'internvl_chat_gpt_oss'])
def vision_module(request):
    root = Path(__file__).resolve().parents[1]
    model_dir = root / request.param / 'internvl/model/internvl_chat'
    # Load the real vision code without importing unrelated chat-model backends.
    package_name = '_test_vision_' + request.param
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    sys.modules[package_name] = package
    try:
        yield importlib.import_module(package_name + '.modeling_intern_vit')
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + '.'):
                del sys.modules[name]


def tiny_config(module, layers=4, drop_path_rate=0.3):
    return module.InternVisionConfig(
        image_size=28, patch_size=14, hidden_size=32,
        intermediate_size=64, num_attention_heads=4,
        num_hidden_layers=layers, drop_path_rate=drop_path_rate,
        use_flash_attn=False,
    )


@pytest.mark.parametrize('device', ['cpu', 'meta'])
@pytest.mark.parametrize('layers,rate', [(1, 0.0), (1, 0.3), (4, 0.0), (4, 0.3)])
def test_vision_initialization(vision_module, device, layers, rate):
    config = tiny_config(vision_module, layers, rate)
    expected = torch.linspace(0, rate, layers, device='cpu').tolist()
    with torch.device(device):
        model = vision_module.InternVisionModel(config)
    parameters = list(model.parameters())
    assert parameters
    assert all(parameter.device.type == device for parameter in parameters)
    for attribute in ('drop_path1', 'drop_path2'):
        actual = [getattr(getattr(layer, attribute), 'drop_prob', 0.0)
                  for layer in model.encoder.layers]
        assert actual == expected


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_meta_checkpoint_round_trip(vision_module, tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    torch.manual_seed(1254)
    model = vision_module.InternVisionModel(tiny_config(vision_module)).to(device).eval()
    pixels = torch.randn(1, 3, 28, 28).to(device)
    with torch.no_grad():
        expected = model(pixel_values=pixels).last_hidden_state.cpu()
    checkpoint = tmp_path / 'vision.pt'
    torch.save(model.state_dict(), checkpoint)
    with torch.device('meta'):
        restored = vision_module.InternVisionModel(tiny_config(vision_module))
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), assign=True)
    restored = restored.to(device).eval()
    assert all(parameter.device.type == device for parameter in restored.parameters())
    assert all(not parameter.is_meta for parameter in restored.parameters())
    with torch.no_grad():
        actual = restored(pixel_values=pixels.to(device)).last_hidden_state.cpu()
    assert actual.shape == (1, 5, 32)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
