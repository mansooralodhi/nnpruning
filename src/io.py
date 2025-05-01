
import torch
import torch.nn as nn

def load_pruned_model(model: nn.Module, file: str) -> None:
    print(f'Loading the pruned model: {file}')
    state = torch.load(file)
    out_features = None
    for layer_param, value in state.items():
        layer_name, attr_name = layer_param.split('.')
        layer = getattr(model, layer_name)
        setattr(layer, attr_name, nn.Parameter(value))
        if attr_name == 'bias':
            '''fc layer in_features/out_features is updated with 'bias '''
            if out_features is not None:
                setattr(layer, 'in_features', out_features)
            setattr(layer, 'out_features', value.shape[0])
            out_features = value.shape[0]
        setattr(model, layer_name, layer)


if __name__ == '__main__':
    from src.mnist.model import SimpleNN

    x = torch.randn((1, 784), dtype=torch.float32)
    file = 'src/mnist/sessions/21-04-2025/one_shot_pruning/0/prune_w_retrain/best_model.pth'
    model = SimpleNN()
    load_pruned_model(model, file)
    tot_p = sum(p.numel() for p in model.parameters())
    print(f'Total parameters in the model: {tot_p}')
