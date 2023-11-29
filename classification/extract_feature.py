import functools
from collections import OrderedDict


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


class IntermediateLayerGetter:
    def __init__(self, model, return_layers, keep_output=True):
        """Wraps a Pytorch module to get intermediate values

        Arguments:
            model {nn.module} -- The Pytorch module to call
            return_layers {dict} -- Dictionary with the selected submodules
            to return the output (format: {[current_module_name]: [desired_output_name]},
            current_module_name can be a nested submodule, e.g. submodule1.submodule2.submodule3)

        Keyword Arguments:
            keep_output {bool} -- If True model_output contains the final model's output
            in the other case model_output is None (default: {True})

        Returns:
            (mid_outputs {OrderedDict}, model_output {any}) -- mid_outputs keys are
            your desired_output_name (s) and their values are the returned tensors
            of those submodules (OrderedDict([(desired_output_name,tensor(...)), ...).
            See keep_output argument for model_output description.
            In case a submodule is called more than one time, all it's outputs are
            stored in a list.
        """
        self._model = model
        self.return_layers = return_layers
        self.keep_output = keep_output

    def __call__(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name, new_name in self.return_layers.items():
            layer = rgetattr(self._model, name)

            def hook(module, input, output, new_name=new_name):
                if new_name in ret:
                    if type(ret[new_name]) is list:
                        ret[new_name].append(output)
                    else:
                        ret[new_name] = [ret[new_name], output]
                else:
                    ret[new_name] = output

            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f'Module {name} not found')
            handles.append(h)

        if self.keep_output:
            output = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None

        for h in handles:
            h.remove()

        return ret, output


def main(args, config):
    from models import build_model
    import torchvision.transforms as T
    from PIL import Image

    model = build_model(config)
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()

    # examples:
    # return_layers = {
    #     'patch_embed': 'patch_embed',
    #     'levels.0.downsample': 'levels.0.downsample',
    #     'levels.0.blocks.0.dcn': 'levels.0.blocks.0.dcn',
    # }
    return_layers = {k: k for k in args.keys}
    mid_getter = IntermediateLayerGetter(model, return_layers=return_layers, keep_output=True)

    image = Image.open(args.img)

    transforms = T.Compose([
        T.Resize(config.DATA.IMG_SIZE),
        T.ToTensor(),
        T.Normalize(config.AUG.MEAN, config.AUG.STD)
    ])
    image = transforms(image)
    image = image.unsqueeze(0)
    image = image.cuda()

    mid_outputs, model_output = mid_getter(image)

    for k, v in mid_outputs.items():
        print(k, v.shape)

    return mid_outputs, model_output


if __name__ == '__main__':
    import argparse
    import torch
    from config import get_config

    parser = argparse.ArgumentParser('Get Intermediate Layer Output')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='Path to config file')
    parser.add_argument('--img', type=str, required=True, metavar="FILE", help='Path to img file')
    parser.add_argument("--keys", default=None, nargs='+', help="The intermediate layer's keys you want to save.")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--save', action='store_true', help='Save the results.')
    args = parser.parse_args()
    config = get_config(args)

    mid_outputs, model_output = main(args, config)

    if args.save:
        torch.save(mid_outputs, args.img[:-3] + '.pth')