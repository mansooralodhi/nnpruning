import copy
import torch
import torch.nn as nn
import torch.nn.modules.activation as activation
from src.nnmodulelike import FCLayer, Relu, NNModuleLike

activation_functions = tuple(cls for cls in activation.__dict__.values() if isinstance(cls, type))

mapping = { nn.modules.linear.Linear: FCLayer, 
            nn.modules.activation.ReLU: Relu}
    
def nnModule_to_nnModuleLike(model: nn.Module) -> NNModuleLike:
    layers = list()
    traced = torch.fx.symbolic_trace(model)
    for layer in traced.graph.nodes:
        if layer.op == "call_module":
            t_layer = getattr(model, layer.target)
            layer_type = type(t_layer)
            mapped_layer = mapping[layer_type](layer.name) if layer_type in activation_functions else mapping[layer_type](t_layer, layer.name)
            layers.append(mapped_layer)
    return layers

def nnModuleLike_to_nnModule(layers: NNModuleLike, model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for layer in layers:
                if layer.name == name:
                    module.out_features = layer.weight.T.shape[0]
                    module.in_features = layer.weight.T.shape[1]
                    module.weight.data = layer.weight.T.clone()
                    module.bias.data = layer.bias.T.clone()
                    break
    return model
    



if __name__ == '__main__':
    from src.mnist.model import SimpleNN
    model = SimpleNN()
    pth_file = 'src/mnist/sessions/2025-03-15/05-44-39/best_model.pth'
    model.load_state_dict(torch.load(pth_file))
    
    x = torch.randn((1, 784))
    y = model(x)
    print(y)

    map = MapNN()

    print('CUstom Nodes:')
    nodes = map.nnModule_to_nnModuleLike(model)

    print('Model Layers:')
    updated_model = map.nnModuleLike_to_nnModule(nodes, model)
    y = updated_model(x)
    print(y)


    