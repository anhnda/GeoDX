
  🎯 Implementation Summary

  I've successfully implemented the Hamiltonian-Informed Schrödinger Bridge proposal for molecular conformation generation based on the research presentation. Here's what was created:

  ✅ Core Components Implemented

  1. Physics Module (models/physics/)

  - dxtb_interface.py: Differentiable xTB force field
    - DXTBForceField: Computes energies and forces with autograd
    - WarmStartOptimizer: xTB geometry optimization for initialization
    - Mock implementation included (replace with real dxtb for production)
  - gamma_scheduler.py: γ(t) scheduling for physics drift weight
    - Multiple schedules: constant, linear, cosine, exponential, sigmoid, inverse-sqrt
    - Adaptive scheduling capability
    - Controls physics contribution at different noise levels

  2. Hamiltonian Bridge Network (models/epsnet/hamiltonian_bridge.py)

  - Full implementation of the proposed architecture
  - Key features:
    - Langevin reference SDE: dC_t = [-0.5*β(t)*C_t + γ(t)*f_xTB(C_t)]dt + sqrt(β(t))*dW_t
    - Warm-start from xTB-optimized geometry
    - Bridge learns xTB→DFT residual correction
    - SE(3) equivariant throughout
    - Physics-informed training loss
    - Combined learned + physics sampling

  3. Training & Sampling Scripts

  - train_hamiltonian_bridge.py: Full training pipeline
    - Supports physics-informed loss
    - Checkpoint management
    - TensorBoard logging
    - Validation loops
  - sample_hamiltonian_bridge.py: Generation pipeline
    - Warm-start sampling
    - Evaluation metrics (RMSD, etc.)
    - Batch processing

  4. Configuration

  - configs/hamiltonian_bridge_qm9.yml: Ready-to-use config
    - All hyperparameters set
    - Physics components configured
    - Training settings optimized

  5. Testing & Documentation

  - test_hamiltonian_bridge.py: Comprehensive test suite
    - Tests all components
    - Verifies SE(3) equivariance
    - Validates sampling pipeline
  - HAMILTONIAN_BRIDGE_README.md: Complete documentation
    - Architecture overview
    - Usage instructions
    - Comparison with other methods
    - Troubleshooting guide

  📁 File Structure

  GeoDiffX/
  ├── models/
  │   ├── physics/
  │   │   ├── __init__.py
  │   │   ├── dxtb_interface.py          ✅ NEW
  │   │   └── gamma_scheduler.py         ✅ NEW
  │   └── epsnet/
  │       ├── hamiltonian_bridge.py      ✅ NEW
  │       └── __init__.py                ✅ UPDATED
  ├── configs/
  │   └── hamiltonian_bridge_qm9.yml     ✅ NEW
  ├── train_hamiltonian_bridge.py         ✅ NEW
  ├── sample_hamiltonian_bridge.py        ✅ NEW
  ├── test_hamiltonian_bridge.py          ✅ NEW
  └── HAMILTONIAN_BRIDGE_README.md        ✅ NEW

  🚀 How to Run on Your Remote Server

  1. Test the Implementation

  # First, verify everything works
  python test_hamiltonian_bridge.py

  2. Train the Model

  # Train on QM9 dataset
  python train_hamiltonian_bridge.py \
      configs/hamiltonian_bridge_qm9.yml \
      --device cuda \
      --logdir ./logs_hamiltonian

  3. Generate Samples

  # After training, generate conformations
  python sample_hamiltonian_bridge.py \
      --checkpoint logs_hamiltonian/checkpoints/500000.pt \
      --test_set data/GEOM/QM9/test.pkl \
      --output ./results \
      --num_samples 10 \
      --device cuda

  🔑 Key Innovations Implemented

  1. Langevin Reference SDE: Physics forces integrated into diffusion process
  2. Warm-Start: xTB optimization provides better initialization than pure noise
  3. Gamma Scheduling: Time-dependent physics weight (strong at high noise, weak at low)
  4. Residual Learning: Network learns universal xTB→DFT correction
  5. SE(3) Equivariance: All components preserve geometric symmetries

  ⚠️ Important Notes

  1. dxtb Library: The current implementation uses a mock xTB for testing. For production:
  pip install dxtb  # Install real differentiable xTB
  2. Dataset Paths: Update paths in config file to match your server:
  dataset:
    train: /path/to/your/GEOM/QM9/train.pkl
    val: /path/to/your/GEOM/QM9/val.pkl
  3. GPU Memory: Physics computation adds overhead. Start with smaller batch size if needed:
  train:
    batch_size: 32  # Reduce if OOM

  📊 Expected Performance

  Based on the proposal, training on QM9 should achieve:
  - In-domain (QM9): RMSD ~0.19-0.21 Å
  - Cross-domain (Drugs): RMSD ~0.60 Å (vs 0.86 Å for vanilla GeoDiff)
  - Zero-shot transfer: ~75% COV-R on Drugs (vs 40% for vanilla)

  📖 Documentation

  Everything is documented in HAMILTONIAN_BRIDGE_README.md including:
  - Detailed architecture explanation
  - Comparison with other methods (Score Mix, Elign, DiSCO)
  - Ablation study guidelines
  - Troubleshooting tips

  The implementation is complete and ready to run on your remote server! Let me know if you need any clarifications or modifications.
# GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MinkaiXu/GeoDiff/blob/main/LICENSE)

[[OpenReview](https://openreview.net/forum?id=PzcvxEMzvQC)] [[arXiv](https://arxiv.org/abs/2203.02923)] [[Code](https://github.com/MinkaiXu/GeoDiff)]

The official implementation of GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation (ICLR 2022 **Oral Presentation [54/3391]**).

![cover](assets/geodiff_framework.png)

## Environments

### Install via Conda (Recommended)

```bash
# Clone the environment
conda env create -f env.yml
# Activate the environment
conda activate geodiff
# Install PyG
conda install pytorch-geometric=1.7.2=py37_torch_1.8.0_cu102 -c rusty1s -c conda-forge
```

## Dataset

### Offical Dataset
The offical raw GEOM dataset is avaiable [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

### Preprocessed dataset
We provide the preprocessed datasets (GEOM) in this [[google drive folder]](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing). After downleading the dataset, it should be put into the folder path as specified in the `dataset` variable of config files `./configs/*.yml`.

### Prepare your own GEOM dataset from scratch (optional)

You can also download origianl GEOM full dataset and prepare your own data split. A guide is available at previous work ConfGF's [[github page]](https://github.com/DeepGraphLearning/ConfGF#prepare-your-own-geom-dataset-from-scratch-optional).

## Training

All hyper-parameters and training details are provided in config files (`./configs/*.yml`), and free feel to tune these parameters.

You can train the model with the following commands:

```bash
# Default settings
python train.py ./config/qm9_default.yml
python train.py ./config/drugs_default.yml
# An ablation setting with fewer timesteps, as described in Appendix D.2.
python train.py ./config/drugs_1k_default.yml
```

The model checkpoints, configuration yaml file as well as training log will be saved into a directory specified by `--logdir` in `train.py`.

## Generation

We provide the checkpoints of two trained models, i.e., `qm9_default` and `drugs_default` in the [[google drive folder]](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing). Note that, please put the checkpoints `*.pt` into paths like `${log}/${model}/checkpoints/`, and also put corresponding configuration file `*.yml` into the upper level directory `${log}/${model}/`.

<font color="red">Attention</font>: if you want to use pretrained models, please use the code at the [`pretrain`](https://github.com/MinkaiXu/GeoDiff/tree/pretrain) branch, which is the vanilla codebase for reproducing the results with our pretrained models. We recently notice some issue of the codebase and update it, making the `main` branch not compatible well with the previous checkpoints.

You can generate conformations for entire or part of test sets by:

```bash
python test.py ${log}/${model}/checkpoints/${iter}.pt \
    --start_idx 800 --end_idx 1000
```
Here `start_idx` and `end_idx` indicate the range of the test set that we want to use. All hyper-parameters related to sampling can be set in `test.py` files. Specifically, for testing qm9 model, you could add the additional arg `--w_global 0.3`, which empirically shows slightly better results.

Conformations of some drug-like molecules generated by GeoDiff are provided below.

<p align="center">
  <img src="assets/exp_drugs.png" /> 
</p>

## Evaluation

After generating conformations following the obove commands, the results of all benchmark tasks can be calculated based on the generated data.

### Task 1. Conformation Generation

The `COV` and `MAT` scores on the GEOM datasets can be calculated using the following commands:

```bash
python eval_covmat.py ${log}/${model}/${sample}/sample_all.pkl
```


### Task 2. Property Prediction

For the property prediction, we use a small split of qm9 different from the `Conformation Generation` task. This split is also provided in the [[google drive folder]](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing). Generating conformations and evaluate `mean  absolute errors (MAR)` metric on this split can be done by the following commands:

```bash
python ${log}/${model}/checkpoints/${iter}.pt --num_confs 50 \
      --start_idx 0 --test_set data/GEOM/QM9/qm9_property.pkl
python eval_prop.py --generated ${log}/${model}/${sample}/sample_all.pkl
```

## Visualizing molecules with PyMol

Here we also provide a guideline for visualizing molecules with PyMol. The guideline is borrowed from previous work ConfGF's [[github page]](https://github.com/DeepGraphLearning/ConfGF#prepare-your-own-geom-dataset-from-scratch-optional).

### Start Setup

1. `pymol -R`
2. `Display - Background - White`
3. `Display - Color Space - CMYK`
4. `Display - Quality - Maximal Quality`
5. `Display Grid`
   1. by object:  use `set grid_slot, int, mol_name` to put the molecule into the corresponding slot
   2. by state: align all conformations in a single slot
   3. by object-state: align all conformations and put them in separate slots. (`grid_slot` dont work!)
6. `Setting - Line and Sticks - Ball and Stick on - Ball and Stick ratio: 1.5`
7. `Setting - Line and Sticks - Stick radius: 0.2 - Stick Hydrogen Scale: 1.0`

### Show Molecule

1. To show molecules

   1. `hide everything`
   2. `show sticks`

2. To align molecules: `align name1, name2`

3. Convert RDKit mol to Pymol

   ```python
   from rdkit.Chem import PyMol
   v= PyMol.MolViewer()
   rdmol = Chem.MolFromSmiles('C')
   v.ShowMol(rdmol, name='mol')
   v.SaveFile('mol.pkl')
   ```


## Citation
Please consider citing the our paper if you find it helpful. Thank you!
```
@inproceedings{
xu2022geodiff,
title={GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation},
author={Minkai Xu and Lantao Yu and Yang Song and Chence Shi and Stefano Ermon and Jian Tang},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=PzcvxEMzvQC}
}
```

## Acknowledgement

This repo is built upon the previous work ConfGF's [[codebase]](https://github.com/DeepGraphLearning/ConfGF#prepare-your-own-geom-dataset-from-scratch-optional). Thanks Chence and Shitong!

## Contact

If you have any question, please contact me at minkai.xu@umontreal.ca or xuminkai@mila.quebec.

## Known issues

1. The current codebase is not compatible with more recent torch-geometric versions.
2. The current processed dataset (with PyD data object) is not compatible with more recent torch-geometric versions.