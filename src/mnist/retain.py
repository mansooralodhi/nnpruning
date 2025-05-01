
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 

class GeneratePruneData:

    def __init__(self, dir: Path = None):
        self.q =  .8
        self.h1, self.h2 = 256, 64  
        self.dir = Path(__file__).parent  / 'retain' if dir is None else dir 
        self.dir.mkdir(parents=True, exist_ok=True)
        self.iterations = list(range(0,31))
        self.data = []
        self.headers = ['Iteration', 'h1NodesRetained', 
                        'h2NodesRetained', 'perNodesRetained']

    def generate_data(self, save_data: bool = True):
        N1, N2 = self.h1, self.h2
        # self.data.append([1, self.h1, self.h2, 100.0])
        for t in self.iterations:
            t1 = round(self.h1 * (self.q) ** t) 
            t2 = round(self.h2 * (self.q) ** t)
            N1 = t1 if t1 > 1 else 1
            N2 = t2 if t2 > 1 else 1
            if t>0 and self.data[-1][1] == N1 and self.data[-1][2] == N2:
                continue 
            per = (N1 + N2) / (self.h1+self.h2) * 100 
            self.data.append([t, N1, N2, per])
            print(f'Iteraiton: {t}')
            print(f'% retained {per}')
            print(f'self.h1 nodes retained {N1}, self.h2 nodes retained {N2}')
            if N1 == N2 == 1:
                break
        if save_data:
            file = self.dir / 'retain.txt'
            print(f'Saving file: {file.as_posix()}') 
            np.savetxt(file.as_posix(), self.data, header=' '.join(self.headers), fmt='%g', comments='')
        return self.data

    def plot_data_distribution(self):
        data = np.asarray(self.data)
        x = data[:, 1] 
        y = data[:, 2]
        plt.scatter(x, y, color='k', label='Retained Neurons Per Layer')
        plt.xlabel('Hidden Layer 1'); plt.ylabel('Hidden Layer 2')
        plt.title('Neuron Retention Across Hidden Layers During Pruning')
        plt.legend(loc='lower right')
        plt.grid(); plt.gca().invert_xaxis();  plt.gca().invert_yaxis()
        file = self.dir / 'pruneNodesDistribution.png' 
        plt.savefig(file.as_posix()); plt.show()
    
    def plot_data_iterations(self):
        data = np.asarray(self.data)
        x = list(range(1, data.shape[0]+1))
        y1 = data[:, 1]
        y2 = data[:, 2]
        plt.plot(x, y1, marker='.', linestyle='--', label="Hidden Layer 1 Neurons", color='#000000')
        plt.plot(x, y2, marker='.', linestyle='--', label="Hidden Layer 2 Neurons", color='#FF7F0E')
        plt.xlabel('Iteration'); plt.ylabel('Number of Neurons')
        plt.title('Neuron Retention Across Hidden Layers During Pruning')
        plt.legend(loc='upper right'); plt.grid()
        plt.gca().set_xticks(range(0, x[-1], 2))
        plt.gca().set_yticks(range(0, self.h1, 20))
        file = self.dir / 'pruneNodesIteration.png' 
        plt.savefig(file); plt.show()

    
if __name__ == '__main__':
    gen = GeneratePruneData()
    gen.generate_data()
    # gen.plot_data_distribution()
    gen.plot_data_iterations()