
**New Feature!** We have now added an unofficial reimplementation of [CircuitSAT](https://openreview.net/forum?id=BJxgz2R9t7) by Amizadeh et al. to our repository, including training and evaluation with the **SAT attacks**.

# Generalization of Neural Combinatorial Solvers Through the Lens of Adversarial Robustness

# Repository Information

This repository contains the official implementation of the ICLR 2022 paper "Generalization of Neural Combinatorial Solvers Through the Lens of Adversarial Robustness" by Simon Geisler & Johanna Sommer, Jan Schuchardt, Aleksandar Bojchevski and Stephan Günnemann. It provides a framework to evaluate the adversarial robustness and generalization capabilities of neural combinatorial solvers for SAT and TSP. </br>

Abstract: </br>
End-to-end (geometric) deep learning has seen first successes in approximating the solution of combinatorial optimization problems. However, generating data in the realm of NP-hard/-complete tasks brings practical and theoretical challenges, resulting in evaluation protocols that are too optimistic. Specifically, most datasets only capture a simpler subproblem and likely suffer from spurious features. We investigate these effects by studying adversarial robustness - a local generalization property - to reveal hard, model-specific instances and spurious features. For this purpose, we derive perturbation models for SAT and TSP. Unlike in other applications, where perturbation models are designed around subjective notions of imperceptibility, our perturbation models are efficient and sound, allowing us to determine the true label of perturbed samples without a solver. Surprisingly, with such perturbations, a sufficiently expressive neural solver does not suffer from the limitations of the accuracy-robustness trade-off common in supervised learning. Although such robust solvers exist, we show empirically that the assessed neural solvers do not generalize well w.r.t. small perturbations of the problem instance. </br>

This repository contains code snippets from the following repositories: </br>
- [TSP-GNN](https://github.com/machine-reasoning-ufrgs/TSP-GNN) by Marcelo de Oliveira Rosa Prates, Rafael Baldasso Audibert and Pedro Henrique da Costa Avelar
- [NeuroSAT](https://github.com/dselsam/neurosat) by Daniel Selsam
- [Graph Convnet TSP](https://github.com/chaitjo/graph-convnet-tsp) by Chaitanya Joshi

We thank the authors of these repositories for making their code public as well as the teams behind [_pysat_](https://github.com/pysathq/pysat), [_python-concorde_](https://github.com/jvkersch/pyconcorde) and [_seml_](https://github.com/TUM-DAML/seml) for their work on the respective packages.

# Installation
You can install all required packages and create a conda environment by running 
```bash
conda create --name <env> --file requirements.txt
```
Please note that we use CUDA version 11.4 and PyTorch version 1.10. We can not guarantee that our code works out of the box with different CUDA/PyTorch versions. </br>

Next, you can install the SAT and TSP packages via
```bash
cd sat
pip install -e .
cd ../tsp/
pip install -e .
```

Lastly, you will have to install Pyconcorde to generate the TSP data. To do that, clone the repository and build it like this: 
```bash
git clone https://github.com/jvkersch/pyconcorde
cd pyconcorde
pip install -e .
```

# SAT
## Experiments
First, you have to generate the SAT dataset. To generate the SAT-10-40 dataset, which is used for training the NeuroSAT model, run: </br>
```bash
python sat/data/gen/generate_nsat.py
```

To train a single NeuroSAT model, run: </br>
```bash
python sat/scripts/train.py
```

This will save the best model to the specified directory and print the training statistics. To do additional adversarial finetuning, run: </br>
```bash
python sat/scripts/train.py -advtrain
```

To run attacks on a model, run:</br>
```bash
python sat/scripts/attacks.py -type sat
python sat/scripts/attacks.py -type del
python sat/scripts/attacks.py -type adc
```
for random attacks and </br>
```bash
python sat/scripts/attacks.py -type sat -opt
python sat/scripts/attacks.py -type del -opt
python sat/scripts/attacks.py -type adc -opt
```
for optimized attacks. You can add `-model_path /path/to/your/model.pt` to any of the attack script settings to specify a specific trained model.

## Q&A

### How do I reproduce the trained models that were attacked for the paper experiments?
To reproduce this, you can find the *SEML* scripts at `scripts/reproduce_train.yaml` and `scripts/reproduce_train.py`. If you do not want to use *SEML*, you can find the exact random seeds used to intitialize the model in the .yaml file and enter them into the training script manually.

### Where can I find the other SAT datasets that you used?
The datasets SAT-3-10, SAT-50-100, SAT-100-300 are generated by modifying the 'min_n' and 'max_n' parameters of the data generation script `data/gen/generate_nsat.py`. The SATLIB and uni3sat data can be found [here](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html).

### How do I convert other SAT data into your data format and train models / run attacks on?
Under `data/gen/parse_sat.py` you can find a script that takes as input the DIMACS CNF format and parses it into .pkl files that contains all information our framework needs. You can then simply add the dataset to `sat/data.get_NSAT_training_data()` and `sat/data.get_NSAT_val_data()` to train models and run attacks.

### I want to integrate a different Neural SAT solver, what do I do?
No problem! When you build your new neural solver, make sure that it implements the functions `reconstruct`, `get_representation` and `collate_fn`. `reconstruct` and `get_representation` should be fully differentiable functions that convert the data representation of the SAT problem that your model expects to the data representation that our attacks operate on. Our representation is defined as follows: a problem x is represented as a CNF with m clauses and n variables. We can then represent X as an indicator matrix $\mathbb{I} \in \mathbb{R}^{2n \times m}$, where the first n rows represent the positive literal and the rows n-2n represent the negative literal of their respective variable and the rows represent the m clauses. The entry $\mathbb{I}_{i, j}$ is 1 if literal i is present in clause j and 0 otherwise. As we would like to work in a scenario where we process multiple samples at the same time in form of a batch, we created a batched indicator $\mathcal{I}$ by placing the individual indicators of a sample on the diagonal of $\mathcal{I}$. Additionally, the attacks need access to the number of variables and the number of clauses present in the individual problem, so make sure to include those into your collate function.

### How can I inspect the perturbation that the adversarial attack causes?
The `tests/` directory provides you with two tools to better understand what the adversarial attacks are doing to the problem. Run the notebook `vis_perturbations.ipynb` to get plots for the clean adjacency, the perturbed adjacency and the differences in the adjacencies of the graph. `tests_neurosat.py` provides you with pytests to verify that the perturbed sample did infact retain its label. You can use the functionality used in these tests to get the perturbed samples in CNF format.

# TSP
## Experiments
First, you have to generate the TSP dataset. To do this, run: </br>
```bash
python tsp/data/generate_tsp.py -path 'TSP/'
```

To train a DecisionTSP model, run: </br>
```bash
python tsp/scripts/train_dtsp.py
```
To train a ConvTSP model, run: </br>
```bash
python tsp/scripts/train_ctsp.py
```

To run random / optimized attacks on a model, run:</br>
```bash
python tsp/scripts/attack_dtsp.py 
python tsp/scripts/attack_dtsp.py -opt
python tsp/scripts/attack_ctsp.py 
python tsp/scripts/attack_ctsp.py -opt
```
You can add `-model_path /path/to/your/model.pt` to any of the attack script settings to specify a specific trained model.

## Q&A

### How can I generate other TSP datasets?
You can refer to the data generation script `tsp/data/generate_tsp.py` and exchange the random uniform sampling in line 72 by any coordinate generation strategy you like. 

### I want to integrate a different Neural TSP solver, what do I do?
Similar to the SAT case, make sure that your model implements the functions `reconstruct` and `collate_fn`. The TSP attack operates on the coordinates of the TSP problem, which means that your collate function has to store the coordinates of each sample and your reconstruction functionality has to build a graph (plus possible features, labels, etc.) from these coordinates.

