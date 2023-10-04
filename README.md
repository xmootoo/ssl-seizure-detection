# Seizure Detection using Graph Neural Networks and Self-Supervised Learning
<img src="https://iiif.elifesciences.org/journal-cms/subjects%2F2021-11%2Felife-sciences-neuroscience-illustration.jpg/0,1,7016,2081/2000,/0/default.jpg" alt="Project Image" width="1000" height="300">

## Introduction

This project utilizes self-supervised learning (SSL) Graph Neural Network (GNN) models for the downstream task of seizure detection. Our project uses a specialized GNN encoder adapted to SSL methods including Relative Positioning, Temporal Shuffling, Contrastive Predictive Coding (CPC), and Variance-Invariance-Covariance Regularization (VICReg). The GNN architecture comprises of Edge-Conditioned Convolution (ECC) and Graph Attention Network (GAT) layers, using the PyTorch and PyTorch Geometric libraries for standard deep learning and GNN implementation. For a detailed guide description, please see: https://www.xaviermootoo.com/projects/ssl-seizure-detection

> Please refer to the relevant papers:
> - Edge-Conditioned Convolution (ECC): [(Simonovsky & Komodakis, 2017)](https://arxiv.org/abs/1704.02901)
> - Graph Attention Layer (GAT): [(Veličković et al., 2018)](https://arxiv.org/abs/1710.10903)
> - Relative Positioning and Temporal Shuffling [(Banvile et al., 2021)](https://arxiv.org/abs/2007.16104)
> - Contrastive Predictive Coding (CPC) [(Oord et al., 2018)](https://arxiv.org/abs/1807.03748)
> - Variance-Invariance-Covariance Regularization (VICreg) [(Bardes et al., 2022)](https://arxiv.org/abs/2105.04906)

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [File Descriptions](#file-descriptions)
4. [License](#license)
5. [Contact](#contact)

## Installation

### Prerequisites
- Python 3.8-3.11
- PyTorch 2.0.1+
- PyTorch Geometric (PyG)

### Steps
Clone the repository: `git clone https://github.com/yourusername/seizure-detection-gnn.git`

## Usage

To run the main program, use the following command (which we optimized for GPU usage on the Graham cluster in Digital Research Alliance of Canada (Canada Compute).

```bash
python $data_path $model_path $stats_path $model_name $num_workers main.py
```

**A)**  

**B)** For information on preprocessing, please see: `preprocess.ipynb`.

**C)** For an introductory tutorial to graph pair classification with PyG, please see `tutorial.ipynb`.

## File Descriptions

- **models.py**: Contains both self-supervised and supervised models, including the GNN architecture with ECC and GAT layers.
- **train.py**: Implements the main training loop for both self-supervised and supervised models. Also includes automatic mixed precision, which provides faster training without sacrificing accuracy.
- **tutorial.ipynb**: A Jupyter notebook tutorial on how to use PyTorch Geometric (PyG) in the context of the project.
- **main.py**: The primary script to run the entire pipeline, optimized for Graham cluster resources.
- **preprocess.py**: Includes helper functions for all preprocessing tasks, such as converting initial graph representations to PyG-compatible structures.
- **preprocess.ipynb**: A guided notebook that demonstrates how to use preprocessing functions for both supervised and self-supervised learning.
- **transfer.ipynb**: A notebook illustrating how to apply transfer learning from self-supervised to supervised models and fine-tuning them.

## License

This project is licensed under the MIT License.

## Contact

For any queries, please contact [xmotoo at gmail dot com](mailto:xmootoo@gmail.com).
