# %%
import sys
print(sys.executable)

# %% [markdown]
# # Tutorial 1: Echo State Networks for Text Classification

# %% [markdown]
# This notebook presents a use case example of the ``EsnTorch`` library. It describes the implementation of an **Echo State Network (ESN)** for text classification on the **TREC-6** dataset (question classification).
# 
# The instantiation, training and evaluation of an ESN for text classification is achieved via the following steps:
# 1. Import libraries and modules
# 2. Load and prepare data
# 3. Instantiate the model:
#     - specify parameters
#     - specify learning algorithm
#     - warm up
# 4. Train
# 5. Evaluate

# %% [markdown]
# ## Librairies

# %%
# !pip install transformers==4.8.2
# !pip install datasets==1.7.0
# !pip install tqdm
# !pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# %%
# To enable progress bars in jupyter:

# pip install ipywidgets
# jupyter nbextension enable --py widgetsnbextension
# conda install -c conda-forge nodejs
# jupyter labextension install @jupyter-widgets/jupyterlab-manager

# %%
# Comment this if library is installed!
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

# %%
from tqdm.notebook import tqdm_notebook
from sklearn.metrics import classification_report

import torch

from datasets import load_dataset, Dataset, concatenate_datasets

from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

import esntorch.core.reservoir as res
import esntorch.core.learning_algo as la
import esntorch.core.pooling_strategy as ps
import esntorch.core.esn as esn

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# Set device (cpu or gpu if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %% [markdown]
# ## Load and prepare data

# %% [markdown]
# ### Load and tokenize data

# %%
# Custom functions for loading and preparing data
def tokenize(sample):
    """Tokenize sample"""
    
    sample = tokenizer(sample['text'], truncation=True, padding=False, return_length=True)
    
    return sample
    
def load_and_prepare_dataset(dataset_name, split, cache_dir):
    """
    Load dataset from the datasets library of HuggingFace.
    Tokenize and add length.
    """
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=split, cache_dir=CACHE_DIR)
    
    # Rename label column for tokenization purposes (use 'label-fine' for fine-grained labels)
    dataset = dataset.rename_column('label-coarse', 'labels')
    
    # Tokenize data
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column('length', 'lengths')
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'lengths'])
    
    return dataset

# %%
# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load and prepare data
CACHE_DIR = 'cache_dir/' # put your path here

full_dataset = load_and_prepare_dataset('trec', split=None, cache_dir=CACHE_DIR)
train_dataset = full_dataset['train'].sort("lengths")
test_dataset = full_dataset['test'].sort("lengths")

# Create dict of all datasets
dataset_d = {
    'train': train_dataset,
    'test': test_dataset
    }

# %%
print(dataset_d)

# %% [markdown]
# ### Create dataloaders

# %%
# Create dict of dataloaders

dataloader_d = {}

for k, v in dataset_d.items():
    dataloader_d[k] = torch.utils.data.DataLoader(v, batch_size=256, 
                                                  collate_fn=DataCollatorWithPadding(tokenizer))

# %%
print(dataloader_d)

# %% [markdown]
# ## Instanciate the model

# %% [markdown]
# ### Parameters

# %% [markdown]
# Most parameters are self-explanatory for a reader familiar with ESNs. Please refer to the documentation for further details. 
# 
# The ``mode`` parameter represents the type of reservoir to be considered:
# - ``mode = recurrent_layer``: implements a **classical recurrent reservoir**, specified among others by its ``dim``, ``sparsity``, ``spectral_radius``, ``leaking_rate`` and ``activation_function``.
# - ``mode = linear_layer``: implements a simple **linear layer** specified by its ``dim`` and ``activation_function``.
# - ``mode = no_layer``: implements **the absence of reservoir**, meaning that the embedded inputs are directly fed to the the learning algorithms.
# 
# The comparison between the ``recurrent_layer`` and the ``no_layer``modes enables to assess the impact of the reservoir on the results. The comparison between the ``recurrent_layer`` and the ``linear_layer``modes allows to assess the importance of the recurrence of the reservoir on the results.

# %%
# ESN parameters
esn_params = {
            'embedding': 'bert-base-uncased', # name of Hugging Face model
            'dim': 1000,
            'sparsity': 0.9,
            'spectral_radius': 0.9,
            'leaking_rate': 0.5,
            'activation_function': 'tanh', # 'tanh', 'relu'
            'bias_scaling': 0.1,
            'input_scaling': 0.1,
            'mean': 0.0,
            'std': 1.0,     
            'learning_algo': None,         # initialzed below
            'criterion': None,             # initialzed below (only for learning algos trained with SGD)
            'optimizer': None,             # initialzed below (only for learning algos trained with SGD)
            'pooling_strategy': 'mean',    # 'mean', 'last', None
            'bidirectional': False,        # True, False
            'mode' : 'recurrent_layer',    # 'no_layer', 'linear_layer', 'recurrent_layer'
            'device': device,  
            'seed': 42
            }

# Instantiate the ESN
ESN = esn.EchoStateNetwork(**esn_params)

# %% [markdown]
# ### Learning algorithm
# Choose your learning algo by un-commenting its associated cell.

# %% [markdown]
# The following algorithms are trained via a **direct method**.<br>
# Hence, there is no need to specify any *criterion* and *optimizer*.
# - ``RidgeRegression`` (our implementation)
# - ``RidgeRegression_skl`` (from scikit-learn)
# - ``LinearSVC`` (from scikit-learn)
# - ``LogisticRegression_skl`` (from scikit-learn)

# %%
ESN.learning_algo = la.RidgeRegression(alpha=10.0)

# %%
# ESN.learning_algo = la.RidgeRegression_skl(alpha=10.0)

# %%
# ESN.learning_algo = la.LinearSVC(C=1.0)

# %%
# ESN.learning_algo = la.LogisticRegression_skl()

# %% [markdown]
# The following algorithms are trained via a **gradient descent**.<br>
# Accordingly, a *criterion* and an *optimizer* must be specified.
# - ``LogisticRegression`` (our implementation)
# - ``DeepNN`` (our implementation)

# %%
# if esn_params['mode'] == 'no_layer':
#     input_dim = ESN.layer.input_dim
# else:
#     input_dim = ESN.layer.dim

# if esn_params['bidirectional']:
#     input_dim *= 2 

# %%
# ESN.learning_algo = la.LogisticRegression(input_dim=input_dim, output_dim=6)

# %%
# ESN.learning_algo = la.DeepNN([input_dim, 512, 256, 6])

# %%
# # Needs criterion and otpimizer

# ESN.criterion = torch.nn.CrossEntropyLoss()
# ESN.optimizer = torch.optim.Adam(ESN.learning_algo.parameters(), lr=0.01)

# %% [markdown]
# ### Warm up

# %%
# Put model on device
ESN = ESN.to(device)

# %%
# Warm up the ESN on multiple sentences
if isinstance(ESN.layer, res.LayerRecurrent):
    ESN.warm_up(dataset_d['train'].select(range(10)))

# %% [markdown]
# ## Train

# %% [markdown]
# For **direct methods**, the parameters ``epochs`` and ``iter_steps`` are ignored. 

# %%
ESN.fit(dataloader_d["train"], epochs=3, iter_steps=50)

# %% [markdown]
# ## Evaluate

# %%
# Train predictions and accuracy
train_pred, train_acc = ESN.predict(dataloader_d["train"], verbose=False)
train_acc

# %%
# Test predictions
test_pred, test_acc = ESN.predict(dataloader_d["test"], verbose=False)
test_acc

# %%
# Test classification report
print(classification_report(test_pred.tolist(), 
                            dataset_d['test']['labels'].tolist(), 
                            digits=4))

# %%



