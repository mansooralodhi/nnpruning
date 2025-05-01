
import torch
import torch.nn as nn
# nn.Linear transpose the weight matrix, unlike other neural

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256, bias=True)
        self.fc2 = nn.Linear(256, 64, bias=True)  
        self.fc3 = nn.Linear(64, 10, bias=True)
        self.relu = nn.ReLU()               
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    import torch.fx
    model = SimpleNN()
    traced = torch.fx.symbolic_trace(model)
    print(traced.graph)
    x = torch.randn((10, 784))
    y = model(x)
    print(y.shape)