import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import Sequence
from src.ad import AD  
from src.interval import Interval
from src.io import load_pruned_model
from src.prune import prune_nodes_layerwise
from src.map import nnModule_to_nnModuleLike, nnModuleLike_to_nnModule


class IterativePruning:
    
    def __init__(self, traindataloader, testdataloader, trainer, validator):
        self.trainer = trainer
        self.validator = validator
        self.test_dataloader = testdataloader
        self.train_dataloader = traindataloader
        self.x_interval = self.domain_interval()

    def domain_interval(self):
        X = self.train_dataloader.dataset.train_data.flatten(start_dim=1).float() # (60000, 784)
        lb, ub = torch.min(X, dim=0).values.unsqueeze(0), torch.max(X, dim=0).values.unsqueeze(0)   # [(1, 784), (1, 784)]
        return Interval(lb, ub)

    def prune(self, model: nn.Module, retainNodes: Sequence[int], 
              percentageRetain: float, retrain: bool, log_dir: Path):
        custom_model = nnModule_to_nnModuleLike(model)
        _, _, custom_model = AD.backward(self.x_interval, custom_model)
        pruned_model = prune_nodes_layerwise(custom_model, retainNodes)          
        pruned_model = nnModuleLike_to_nnModule(pruned_model, model)
        if retrain:
            pruned_model = self.trainer(pruned_model, self.train_dataloader, self.test_dataloader, logdir = log_dir)
        else:
            torch.save(pruned_model.state_dict(), log_dir / f'best_model.pth')
        results = self.validator(pruned_model, self.train_dataloader, self.test_dataloader)
        self.save_validation_results(results, percentageRetain, log_dir)
        
    @staticmethod
    def save_validation_results(results, percentRetain, file):
        file = file / 'validation_results.txt'
        results['PercentageRetain'] = percentRetain
        cols = list(results.keys())
        data = np.column_stack([results[col] for col in cols])
        np.savetxt(file , data, header=' '.join(cols), fmt='%g', comments='')


if __name__ == '__main__':
    from src.mnist.model import SimpleNN
    from src.trainer import train
    from src.validator import evaluate
    from src.mnist.dataloader import train_loader, test_loader
    from src.mnist.retain import GeneratePruneData

    model = SimpleNN()
    data = GeneratePruneData().generate_data()
    experiment = IterativePruning(train_loader, test_loader, train, evaluate)

    model_file = ''
    my_dir = Path('mnist/sessions/2025-04-21/iterative_pruning/')
    my_dir.mkdir(exist_ok=True, parents=True)
    my_dir = my_dir.as_posix()

    begin_itr = 9
    for i in range(begin_itr, len(data)):
        print(f'******  Iteration : {i} **********')
        _, h1, h2, per = data[i]
        log_dir = Path(my_dir +  f'{i}/prune_w_retrain')
        log_dir.mkdir(exist_ok=True, parents=True)
        if i == 0:
            train(model, train_loader, test_loader, logdir=log_dir)
        else:
            model_file =  my_dir + f'{i-1}/prune_w_retrain/best_model.pth'
            load_pruned_model(model, model_file)
            experiment.prune(model, [h1, h2], per, False, log_dir)
