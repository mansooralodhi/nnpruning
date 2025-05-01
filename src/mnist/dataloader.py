
import torch
from torchvision import datasets, transforms

def transform(x):
    x = transforms.ToTensor()(x) 
    return x.view(-1) / 255


test_dataset = datasets.MNIST(
    root='src/mnist/data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=True)

train_dataset = datasets.MNIST(
    root='src/mnist/data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True)

