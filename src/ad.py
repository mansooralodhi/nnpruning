import torch
from typing import Sequence

from src.interval import Interval
from src.map import nnModule_to_nnModuleLike
from src.nnModuleLike import NNModuleLike, TensorOrInterval

class AD:

    @classmethod
    def forward_pass(cls, x: TensorOrInterval, model: NNModuleLike) -> TensorOrInterval:
        for layer in model: 
            x = layer.forward(x)
            # print(layer.name)
            # print(x)
        return x, model

    @classmethod
    def backward_pass(cls, x: TensorOrInterval, model: NNModuleLike) -> TensorOrInterval:
        input_adjoint = torch.full((model[-1].primal.shape), 1.0)
        if isinstance(x, Interval): 
            input_adjoint = Interval(input_adjoint, input_adjoint)
            print('output adjoint')
            print(input_adjoint)
        for i in reversed(range(len(model))): 
            if not model[i].grad:
                print(model[i].name)
                print(input_adjoint)
            input_adjoint = model[i].backward(input_adjoint)
            if model[i].grad:
                print(model[i].name)
                print(model[i].adjoint)
            
        return input_adjoint, model
   
    @classmethod
    def backward(cls, x: TensorOrInterval, model: NNModuleLike):
        x_primal, model = cls.forward_pass(x, model)
        x_adjoint, model = cls.backward_pass(x, model)
        return x_primal, x_adjoint, model

    
if __name__ == '__main__':

    from src.mnist.model import SimpleNN

    x = torch.randn((1, 784), dtype=torch.float32)
    xIval = Interval(x-1, x + 1)
    m = SimpleNN()
    m = nnModule_to_nnModuleLike(m)
    primal, adjoint, model = AD.backward(xIval, m)
    print(primal.shape)
    print(adjoint.shape)
    for layer in model:
        if layer.grad:
            print(f'weight shape : {layer.weight.shape}')
            print(f'adjoint shape : {layer.adjoint.shape}')
