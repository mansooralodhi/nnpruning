
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import Sequence
from src.ad import AD  
from src.interval import Interval
from src.nnmodulelike import NNModuleLike
from src.prune import prune_nodes_layerwise
from src.map import nnModule_to_nnModuleLike, nnModuleLike_to_nnModule

class OneShotPruning:
    
    def __init__(self, traindataloader, testdataloader, trainer, validator):
        self.trainer = trainer
        self.validator = validator
        self.test_dataloader = testdataloader
        self.train_dataloader = traindataloader
        self.x_interval: Interval = self.domain_interval()
        self.base_model: NNModuleLike = None

    def domain_interval(self):
        X = self.train_dataloader.dataset.train_data.flatten(start_dim=1).float() # (60000, 784)
        lb, ub = torch.min(X, dim=0).values.unsqueeze(0), torch.max(X, dim=0).values.unsqueeze(0)   # [(1, 784), (1, 784)]
        return Interval(lb, ub)

    def base_model_adjoints(self, model: nn.Module):
        custom_model = nnModule_to_nnModuleLike(model)
        _, _, custom_model = AD.backward(self.x_interval, custom_model)
        self.base_model = custom_model

    def prune(self, retainNodes: Sequence[int], percentageRetain: float, 
                    retrain: bool, log_dir: Path):
        pruned_model = prune_nodes_layerwise(self.base_model, retainNodes)          
        pruned_model = nnModuleLike_to_nnModule(pruned_model, model)
        if retrain:
            pruned_model = self.trainer(pruned_model, self.train_dataloader, 
                                        self.test_dataloader, logdir = log_dir)
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
    from src.io import load_pruned_model

    model_file = ''
    # par_dir = Path(__file__).parent
    # my_dir = Path(par_dir.as_posix() + '/mnist/sessions/2025-04-21/one_shot_pruning/')
    my_dir = Path('src/mnist/sessions/2025-04-21/one_shot_pruning/')
    my_dir.mkdir(exist_ok=True, parents=True)
    
    
    model = SimpleNN()
    data = GeneratePruneData(my_dir.parent).generate_data(save_data=False)
    experiment = OneShotPruning(train_loader, test_loader, train, evaluate)

    begin_itr = 7
    for i in range(begin_itr, len(data)):
        print(f'******* Iteration: {i}/{len(data)} *******')
        _, h1, h2, per = data[i]
        log_dir = Path(my_dir /  f'{i}/prune_w_retrain')
        log_dir.mkdir(exist_ok=True, parents=True)
        if i == 0:
            best_model = train(model, train_loader, test_loader, logdir=log_dir)
            experiment.base_model_adjoints(best_model)
        elif i == begin_itr:
            file = my_dir.as_posix() + '/0/prune_w_retrain/best_model.pth'
            best_model = SimpleNN()
            load_pruned_model(best_model, file)
            experiment.base_model_adjoints(best_model)
        else:
            experiment.prune([h1, h2], per, True, log_dir)