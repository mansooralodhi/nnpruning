
<details open> 
<summary> <ins> Model and Data Prerequisite </ins> </summary>

  - [ ] Generate (any) problem directiory, `src/mnist/`, for mnist data classification.
  - [ ] Define datasets and dataloaders in the file `src/mnist/dataloaders.py`
  - [ ] Download data in the directory `src/mnist/data/`
  - [ ] Define neural network model in the file `src/mnist/model.py`
  - [ ] Generate nodes retention data from the file `src/mnist/retain.py`
  - [ ] Ensure the generated direcotry `src/mnist/retain/` has the files
      - [ ] `retain.txt`
      - [ ] `pruneNodesIteration.png`
      - [ ] `pruneNodesDistribution`
  <div align="center">
    <table>
      <tr>
        <td align="center"><img src="src/mnist/retain/pruneNodesDistribution.png" width="300"/><br>
            <div align="center"> <sub>pruneNodesDistribution.png</sub> </div> 
        </td>
        <td><img src="src/mnist/retain/pruneNodesIteration.png" width="300"/><br> 
            <div align="center"> <sub>pruneNodesIteration.png</sub> </div> 
        </td>
      </tr>
    </table>
  </div>
  </details> 

<details open>
<summary> <ins> Perform Iterative Pruning </ins> </summary>
  
  - [ ] Locate `__name__=='__main__' in the file `src/iterativeprune.py`
  - [ ] Define the neural network model
        <br> `model = SimpleNN()`
  - [ ] Generate pruning data that results in file `src/mnist/retain/retain.txt`
        <br> `data = GeneratePruneData().generate_data()`
  - [ ] Initialze the iterative pruning with relevant arugments
        <br> `experiment = IterativePruning(train_loader, test_loader, train, evaluate)`
  - [ ] Define the directory to store experiment results
        <br> `my_dir = Path('mnist/sessions/2025-04-21/iterative_pruning/')`
  - [ ] Define the `begin_itr` variable
      - [ ] 0: start from scratch
      - [ ] ¬ 0 : start from given pruning iteration
  - [ ] Iterative pruning with retraining results in the directory structure.
        <pre> 
        mnist/sessions/2025-04-21/iterative_pruning/
        ├── 0/
        |   ├── prune_w_retrain/
        |   |    └── best_model.pth
        |   |    └── train_results.txt
        ├── 1/
        |   ├── prune_w_retrain/
        |   |    └── best_model.pth
        |   |    └── train_results.txt
        |   |    └── validation_results.txt
        |   ├── ...
        |   ...
        ...
        ├── ***i***/
        |   ├── prune_w_retrain/
        |   |     └── best_model.pth
        |   |     └── train_results.txt
        |   |     └── validation_results.txt
        </pre>
  - [ ] Once iterations halt, execute the file ```src/concatenate.py``` to generate
      - ```mnist/sessions/2025-04-21/iterative_pruning/prune_w_retrain_results/```
          - ```validations.txt```
          - ```accuracies.png```
          - ```losses.png```
  <div align="center">
    <table>
      <tr>
        <td align="center"><img src="src/mnist/retain/pruneNodesDistribution.png" width="300"/><br>
            <div align="center"> <sub>pruneNodesDistribution.png</sub> </div> 
        </td>
        <td><img src="src/mnist/retain/pruneNodesIteration.png" width="300"/><br> 
            <div align="center"> <sub>pruneNodesIteration.png</sub> </div> 
        </td>
      </tr>
    </table>
  </div>
</details>

<details open>
<summary> Test Optimal Pruned Model </summary>
</details>
