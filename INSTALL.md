# Installation

Ensure you have `conda` installed on your system. If not, install from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).

### Step 1: Clone the Repository
```bash
git clone https://github.com/xmootoo/ssl-seizure-detection.git
cd ssl-seizure-detection
```

### Step 2: Create Conda Environment
```bash
conda create --name ssl-seizure-detection python=3.10
conda activate ssl-seizure-detection
```

### Step 3: Install PyTorch
For PC or Linux:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
For Mac:
```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```


### Step 4: Install PyTorch Geometric
```bash
pip install torch-geometric
```

### Step 5: Install Additional Packages
```bash
pip install scikit-learn pandas wandb
```

### Step 6: Login to Weights & Biases
```bash
wandb login
```
