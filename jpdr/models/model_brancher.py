# Based on IntermediateLayerGetter of torchvision/models/_utils.py
from collections import OrderedDict
from copy import deepcopy

from torch import nn
from typing import Dict


COMMON = 'common'
BRANCH1 = 'branch1'
BRANCH2 = 'branch2'


class ModelBrancher(nn.Module):
    """
    Splits up the backbone in branches with the same architecture, but
    untied weights.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        branch_layer: The name of the module where the model is branched.
        num_branches (int): The number of branches.
    """
    def __init__(self, model: nn.Module, return_layers: Dict[str, str],
                 branch_layer: str, num_branches: int = 2) -> None:
        if not set(return_layers).issubset([
                name for name, _ in model.named_children()
        ]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        common_layers = OrderedDict()
        branches = [OrderedDict() for _ in range(num_branches)]
        is_common = True

        for name, module in model.named_children():
            if name == branch_layer:
                is_common = False
            if is_common:
                common_layers[name] = module
            else:
                for branch_layers in branches:
                    branch_layers[name] = deepcopy(module)
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__()

        self.common = nn.ModuleDict(common_layers)
        self.branches = nn.ModuleList([
            nn.ModuleDict(branch_layers) for branch_layers in branches
        ])

        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()

        for name, module in self.common.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[f'common.{out_name}'] = x

        common_x = x

        for i, branch in enumerate(self.branches):
            x = common_x
            for name, module in branch.items():
                x = module(x)
                if name in self.return_layers:
                    out_name = self.return_layers[name]
                    out[f'branch{i+1}.{out_name}'] = x

        return out
