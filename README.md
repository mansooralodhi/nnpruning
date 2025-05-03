

## Table of Contents
- [Prerequisite](#prerequisite)
- [Iterative Pruning](#iterative-pruning)
- [Test Pruning](#test-pruning)
- [Deployment](#deployment)

### Prerequisite
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

### Iterative Pruning  
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
        <td align="center"><img src="src/mnist/sessions/2025-04-21/iterative_pruning/prune_w_retrain_results/accuracies.png" width="300"/><br>
            <div align="center"> <sub>accuracies.png</sub> </div> 
        </td>
        <td><img src="src/mnist/sessions/2025-04-21/iterative_pruning/prune_w_retrain_results/losses.png" width="300"/><br> 
            <div align="center"> <sub>losses.png</sub> </div> 
        </td>
      </tr>
    </table>
  </div>

### Test Pruning  
  - [ ] Given the loss and accuracy graphs, copy the model path
        <br> ```src/mnist/sessions/2025-04-21/iterative_pruning/8/prune_w_retrain/best_model.pth```
  - [ ] Locate ```if __name__=='__main__'``` in the file ```src/validator.py```
  - [ ] Define the model
        <br> ```model = SimpleNN()```
  - [ ] Load pruned model
        <br> ```load_pruned_model(model, pruned_model_file)```
  - [ ] Define the model and paste the model file here to evaluate the pruned model on test data using
        <br> ```evaluate(model, train_loader, test_loader)```
  - [ ] Use the above model file together with ```load_pruned_model()``` function for runtime execution and deployment. 

### Deployment
  - [ ]  Copy the pruned model file
        <br> ```src/mnist/sessions/2025-04-21/iterative_pruning/8/prune_w_retrain/best_model.pth```
  - [ ]  Copy the neural network model file:
        <br> ```src/mnist/model.py```
  - [ ]  Copy the model loader file
        <br> ```src/io```
  - [ ]  Execute the model in 3 steps below
      -  ```model = SimpleNN()```
      -  ```load_pruned_model(model, pruned_model_file)```
      -  ```model(input)```
