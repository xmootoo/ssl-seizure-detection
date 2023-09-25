# Seizure Detection using Graph Neural Networks and Self-Supervised Learning
<img src="https://iiif.elifesciences.org/journal-cms/subjects%2F2021-11%2Felife-sciences-neuroscience-illustration.jpg/0,1,7016,2081/2000,/0/default.jpg" alt="Project Image" width="1000" height="300">

## Introduction

This project aims to leverage the power of Graph Neural Networks (GNNs) for seizure detection. Specifically, it employs various self-supervised learning methods such as Relative Positioning, Temporal Shuffling, Contrastive Predictive Coding (CPC), and Variance-Invariance-Covariance Regularization (VICReg). The architecture involves the use of a single Edge-Conditioned Convolution (ECC) layer and a Graph Attention Layer (GAT).

> Note: For the ECC and GAT layers, please refer to their respective papers:
> - Edge-Conditioned Convolution (ECC): [(Simonovsky & Komodakis, 2017)](https://arxiv.org/abs/1704.02901)
> - Graph Attention Layer (GAT): [(Veličković et al., 2018)](https://arxiv.org/abs/1710.10903)

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
python main.py
```

For a detailed tutorial, refer to `tutorial.ipynb`.

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
