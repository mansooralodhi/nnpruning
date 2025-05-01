import torch
from abc import abstractmethod
from typing import Union, TypeAlias, Iterable
from typing_extensions import Protocol, runtime_checkable

from src.interval import Interval


TensorOrInterval: TypeAlias = Union[torch.Tensor, Interval]

@runtime_checkable
class NNLayerProtocol(Protocol):

    @property
    @abstractmethod
    def grad(self) -> bool: ...

    @abstractmethod
    def forward(self, x: TensorOrInterval): ...
    
    @abstractmethod
    def backward(self, input_adjoint: TensorOrInterval): ...


class Relu(NNLayerProtocol):
    
    @property
    def grad(self):
        return False
    
    def __init__(self, name: str = ''):
        self.primal = None
        self.name = name
    
    def forward(self, x):
        self.primal =  x * (x > 0) 
        return self.primal
    
    def backward(self, input_adjoint):
        return  input_adjoint * (self.primal > 0)
    
    
class FCLayer(NNLayerProtocol):

    @property
    def grad(self):
        return True

    def __init__(self, layer: torch.nn.Linear, name: str = ''):
        ### since torch nn.Linear transpose the weights, 
        ### hence, we reverse it back to standard.
       
        self.name = name
        self.weight = layer.weight.data.T.clone() 
        self.bias = layer.bias.data.T.clone() if layer.bias is not None else None

        self.primal = None
        self.adjoint = None

        ### if pruning connections: 
        self.weight_grad =  torch.zeros_like(self.weight)
        self.bias_grad =  torch.zeros_like(self.bias) if layer.bias is not None else None

    def forward(self, x):
        self.primal = x @ self.weight
        if self.bias is not None:
            self.primal += self.bias
        return self.primal

    def backward(self, input_adjoint, ):
        self.adjoint = input_adjoint
        return input_adjoint @ self.weight.T

    def adjoint_wrt_weight(self, input_primal, input_adjoint):
        ### if pruning connections:
        self.weight_grad += input_primal.T @ input_adjoint
        if self.bias_grad is not None: 
            self.bias_grad += input_adjoint
        return (input_adjoint @ self.weight.T) 


NNModuleLike: TypeAlias = Iterable[NNLayerProtocol]


if __name__ == '__main__':
    _ = Relu()