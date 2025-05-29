
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Dict, List
from mnist.results import results_columns
from matplotlib.ticker import MultipleLocator


class PlotMetrices:
    '''
    Output: train_accuracies.png, train_losses.png
    '''
    
    def __init__(self, result_file: str):
        result_file = Path(result_file)
        if not result_file.exists():
            raise FileExistsError(f"File {result_file} does not exist.")
        self.results_file = result_file 
        self.x: Sequence[float] = []
        self.y: Dict[str, List] = {'TestLoss': [], 'TrainLoss': [], 
                                   'TestAccuracy': [], 'TrainAccuracy': []}

    def plot(self, xlabel = 'Epoch'):
        self.read_text_file()
        loss_file = self.results_file.parent / f'loss.png'
        accu_file = self.results_file.parent / f'accuracy.png'
        self.plot_loss(self.x, self.y, xlabel, save_file=loss_file.as_posix())
        self.plot_accuracy(self.x, self.y, xlabel, save_file=accu_file.as_posix())
                                        
    def read_text_file(self):
        data = np.genfromtxt(self.results_file, names=True)
        columns = data.dtype.names
        results = {}
        for col in columns:
            results[col] = data[col].tolist()
        self.x = np.arange(len(data)).tolist() if not 'RetainNodesPer' in columns else results['RetainNodesPer'] 
        if set(self.y.keys()) - set(results.keys()):
            raise Exception('Required Loss/Accuracy Not Found !')
        self.y = results

    @staticmethod
    def plot_loss(x, y, xlabel, save_file):
        plt.plot(x, y['TestLoss'], label="Test Loss", color='blue')
        plt.plot(x, y['TrainLoss'], label="Train Loss", color='green')
        plt.axvline(x=47, color='k', linestyle='dotted', linewidth=2, label='Optimal Model')
        plt.legend(); plt.grid()
        plt.xlabel(xlabel); plt.ylabel("Loss")
        plt.title("Model Loss")
        if save_file: plt.savefig(save_file)
        plt.show()

    @staticmethod
    def plot_accuracy(x, y, xlabel, save_file):
        plt.plot(x, y['TestAccuracy'], label = 'Test_Accuracy', color='blue')
        plt.plot(x, y['TrainAccuracy'], label = 'Train_Accuracy', color='green')
        plt.axvline(x=47, color='k', linestyle='dotted', linewidth=2, label='Optimal Model')
        plt.legend(); plt.grid()
        plt.xlabel(xlabel); plt.ylabel("Accuracy")
        plt.title("Model Accuracy")
        if save_file: plt.savefig(save_file)
        plt.show()


class PlotPruneIterations:

    def read_text_file(results_file):
        data = np.genfromtxt(results_file, names=True)
        columns = data.dtype.names
        results = {}
        for col in columns:
            results[col] = data[col].tolist()
        return results

    def plot_loss(columns: Sequence[str], data: Dict[str, List]):
        x = []
        yTrain, yTest = [], []
        for col, results in data.items():
            if col == 'PercentageRetain':
                x = results
            elif col == 'Train_Loss':
                yTrain = results
            elif col == 'Test_Loss':
                yTest = results
        plt.plot(x, yTrain, marker='.', linestyle='--', label='Train Loss', color='#000000')
        plt.plot(x, yTest, marker='.', linestyle='--', label='Test Loss', color='#8BC34A')
        plt.legend(); plt.minorticks_on(); plt.grid(which='both', linestyle='-', linewidth=0.5)
        plt.xlabel(f'% of Total Hidden Nodes'); plt.ylabel('Loss'); plt.title('Iterative Pruning Loss')
        plt.gca().invert_xaxis()
        plt.savefig(fname = 'src/mnist/sessions/2025-04-21/iterative_pruning/prune_w_retrain_results/loss.png', dpi=300)
        plt.show(); plt.plot()

    def plot_accuracy(columns: Sequence[str], data: Dict[str, List]):
        x = []
        yTrain, yTest = [], []
        for col, results in data.items():
            if col == 'PercentageRetain':
                x = results
            elif col == 'Train_Accuracy':
                yTrain = results
            elif col == 'Test_Accuracy':
                yTest = results
        plt.plot(x, yTrain, marker='.', linestyle='--', label='Train Accuracy', color='#000000')
        plt.plot(x, yTest, marker='.', linestyle='--', label='Test Accuracy', color='#E60000')
        plt.legend(); plt.minorticks_on(); plt.grid(which='both', linestyle='-', linewidth=0.5)
        plt.xlabel(f'% of Total Hidden Nodes'); plt.ylabel('Accuracy'); plt.title('Iterative Pruning Accuracy')
        plt.gca().invert_xaxis()
        plt.savefig(fname = 'src/mnist/sessions/2025-04-21/iterative_pruning/prune_w_retrain_results/accu.png', dpi=300)
        plt.show(); plt.plot()

        # file = Path(__file__).parent / 'mnist/pruneEffect/itrAccu.png'


if __name__ == '__main__':
    # file = 'src/mnist/sessions/2025-03-22/05-52-45/results_prune_wo_retrain.txt'
    # PlotMetrices(file).plot(filepostfix='prune_wo_retrain', xlabel='q')

    # file = 'src/mnist/sessions/2025-03-23/14-11-23/results_train.txt'
    # PlotMetrices(file).plot(filepostfix='train', xlabel='Epochs')
    results_file = 'src/mnist/sessions/2025-04-21/iterative_pruning/prune_w_retrain_results/validations.txt'
    data = PlotPruneIterations.read_text_file(results_file)
    PlotPruneIterations.plot_accuracy(results_columns, data)
    PlotPruneIterations.plot_loss(results_columns, data)
