import copy
import torch
from typing import Sequence
from src.nnmodulelike import NNModuleLike


def prune_nodes_layerwise(layers: NNModuleLike, retainLayersNodes: Sequence[int]):
    '''
    Constraint: len(retainLayersNodes) == Number of grad=True layers)
    '''
    i = 0
    keep_indices = []
    out_layer = len(layers) - 1
    layers = copy.deepcopy(layers)
    for l in range(len(layers)):
        if layers[l].grad:
            if len(keep_indices)>0:
                layers[l].weight = torch.index_select(layers[l].weight, 0, keep_indices)
            if l == out_layer:
                break
            # old:
            # sig = (layers[l].primal.width * layers[l].adjoint.upper)
            # new:
            # TODO: take absolute of lower and upper bound and select the max
            sig = (layers[l].primal.width * abs(layers[l].adjoint).upper)
            
            keep_indices = torch.topk(sig, retainLayersNodes[i], largest=True)[1]
            keep_indices = keep_indices.flatten()
            layers[l].bias = torch.index_select(layers[l].bias, 0, keep_indices)
            layers[l].weight = torch.index_select(layers[l].weight, 1, keep_indices)
            layers[l].adjoint.lower = torch.index_select(layers[l].adjoint.lower, 1, keep_indices)
            layers[l].adjoint.upper = torch.index_select(layers[l].adjoint.upper, 1, keep_indices)
            layers[l].primal.lower = torch.index_select(layers[l].primal.lower, 1, keep_indices)
            layers[l].primal.upper = torch.index_select(layers[l].primal.upper, 1, keep_indices)
            i += 1
    return layers



