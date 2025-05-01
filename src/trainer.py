import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from src.mnist.model import SimpleNN

'''
tensorboard --logdir=src/mnist/sessions/2025-04-21/one_shot_pruning/
'''

def log_dir():
    logdir = Path(__file__).parent / 'mnist/sessions' / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S") / '1/'
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"{'*' * 50}\n Began Logging at {logdir.relative_to(Path(__file__).parent)} \n{'*' * 50}")
    return logdir


def train(  model: nn.Module, train_loader, test_loader, 
            lr: float = 0.001, num_epochs: int = 300,
            logdir: Path = '',
            save_model: bool = True) -> nn.Module:
    '''
    Args:
        save_model : save best model or not
        logdir : log tensorboard train/test loss and accuracy on each epoch
        results_file : file to store train/test loss and accuracy on each epoch 
    '''

    if not logdir:
        logdir = log_dir()
    
    writer = SummaryWriter(logdir)
    train_results_file = logdir / 'train_results.txt'

    device = 'cpu'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)

    minLoss = float('inf')
    best_model = None
    lltest, aatest = [], []
    lltrain, aatrain = [], []
    patience = 0

    for epoch in range(1, num_epochs):
        print(f'Trainer Epoch:  {epoch}/{num_epochs}')
        if patience > 10:
            break

        l, a = [], []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            l.append(loss.item())
            a.append((torch.argmax(outputs, dim=1) == targets).sum().item() / len(inputs))

        lltrain.append(torch.asarray(l).mean().item())
        aatrain.append(torch.asarray(a).mean().item() * 100)
        writer.add_scalar("Loss/Train", torch.asarray(l).mean().item(), epoch)
        writer.add_scalar("Accuracy/Train", torch.asarray(a).mean().item() * 100, epoch)

        l, a = [], []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            l.append(loss.item())
            a.append((torch.argmax(outputs, dim=1) == targets).sum().item() / len(inputs))
        
        lltest.append(torch.asarray(l).mean().item())
        aatest.append(torch.asarray(a).mean().item() * 100)
        writer.add_scalar("Loss/Test", torch.asarray(l).mean(), epoch)
        writer.add_scalar("Accuracy/Test", torch.asarray(a).mean() * 100, epoch)
    
        if (minLoss - torch.asarray(l).mean()) > 0.0001:
            patience = 0
            print(f"Trainer Saving Best Model at Epoch {epoch}")
            minLoss = torch.asarray(l).mean()     
            if save_model:
                torch.save(model.state_dict(), logdir / f'best_model.pth')
            best_model = copy.deepcopy(model)
        patience = patience + 1

    results_cols = ['TrainLoss', 'TrainAccuracy', 'TestLoss', 'TestAccuracy']
    results_data = np.column_stack((np.array(lltrain), np.array(aatrain), np.array(lltest), np.array(aatest)))
    
    np.savetxt(train_results_file, results_data, header=' '.join(results_cols), fmt='%g', comments='')

    writer.flush()
    return best_model


if __name__ == '__main__':
    from src.mnist.dataloader import train_loader, test_loader
    from src.mnist.model import SimpleNN
    model = SimpleNN()
    train(model, train_loader, test_loader, 
          logdir=Path('src/mnist/sessions/30-03-2025/recurssion/1'))