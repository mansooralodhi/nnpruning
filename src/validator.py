
import torch
import torch.nn as nn
from typing import Sequence, Dict

def evaluate(model: nn.Module, train_dataloader, test_dataloader) -> Sequence[Dict[str, float]]:

    keys = ['Train_Loss', 'Test_Loss', 'Train_Accuracy', 'Test_Accuracy']
    returnDict  = dict.fromkeys(keys)

    device = 'cpu'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    
    correct, losses = 0, []
    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        losses.append(criterion(outputs, targets).item())
        correct += (torch.argmax(outputs, dim=1) == targets).sum().item()

    accuracy = correct / len(test_dataloader.dataset) * 100
    loss = torch.asarray(losses).mean().item()
    returnDict.update({'Test_Loss': loss, 'Test_Accuracy': accuracy})
 
    print(f'Validation Loss: {loss}, Accuracy: {accuracy}')

    correct, losses = 0, []
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        losses.append(criterion(outputs, targets).item())
        correct += (torch.argmax(outputs, dim=1) == targets).sum().item()

    accuracy = correct / len(train_dataloader.dataset) * 100
    loss = torch.asarray(losses).mean().item()
    returnDict.update({'Train_Loss': loss, 'Train_Accuracy': accuracy})

    return returnDict

if __name__ == '__main__':
    from src.mnist.model import SimpleNN
    from src.io import load_pruned_model
    from src.mnist.dataloader import train_loader, test_loader
    model = SimpleNN()
    load_pruned_model(model,'src/mnist/sessions/2025-04-21/iterative_pruning/0/prune_w_retrain/best_model.pth')
    results = evaluate(model, train_loader, test_loader)
    print(results)