

import torch
import torch.nn as nn
from thop import profile
class Profiler:
    
    def model_profile(model: nn.Module, x = torch.randn((1, 784))):
        flops, params = profile(model, inputs=(x,))
        return flops, params
    
    def model_hidden_nodes(model: nn.Module):
        out_features = 0
        num_hidden_neurons = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                num_hidden_neurons += m.out_features
                out_features = m.out_features
        num_hidden_neurons -= out_features          
        return num_hidden_neurons
    
if __name__ == '__main__':
    from src.mnist.model import SimpleNN
    from src.io import load_pruned_model

    file = 'src/mnist/sessions/2025-04-21/iterative_pruning/0/prune_w_retrain/best_model.pth'    
    model = SimpleNN()
    model.load_state_dict(torch.load(file))
    base_flops, base_params = Profiler.model_profile(model)
    base_n = Profiler.model_hidden_nodes(model)
    print(f'FLOPS: {base_flops}, Parameters: {base_params}, Hidden Neurons: {base_n}')

    file = 'src/mnist/sessions/2025-04-21/iterative_pruning/8/prune_w_retrain/best_model.pth'    
    model = SimpleNN()
    load_pruned_model(model, file)
    flops, params = Profiler.model_profile(model)
    n = Profiler.model_hidden_nodes(model)
    print(f'FLOPS: {flops}, Parameters: {params}, Hidden Neurons: {n}')


    print(f'FLOPS: {flops}, Parameters: {(params/base_params)*100}, Hidden Neurons: {n}')
    
