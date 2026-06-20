# %% [markdown]
# # TREC-6
# # Simple Baselines using ``mean`` and ``last`` pooling

# %% [markdown]
# ## Librairies

# %%
# !pip install transformers==4.8.2
# !pip install datasets==1.7.0
# !pip install ax-platform==0.1.20

# %%
print("INICIO DE LA EJECUCIÓN\n")
import os
import sys
sys.path.insert(0, os.path.abspath("../..")) # comment this if library is pip installed

# %%
import io
import re
import pickle
from timeit import default_timer as timer

from tqdm.notebook import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers import BertModel
from transformers.data.data_collator import DataCollatorWithPadding

from ax import optimize
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting

import esntorch.core.reservoir as res
import esntorch.core.learning_algo as la
import esntorch.core.pooling_strategy as ps
import esntorch.core.esn as esn

# %%
from IPython import get_ipython

ip = get_ipython()

if ip is not None:
    ip.run_line_magic('config', 'Completer.use_jedi = False')
    ip.run_line_magic('load_ext', 'autoreload')
    ip.run_line_magic('autoreload', '2')

print("IMPORTS REALIZADOS\n")

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
SEED = 42

# %%


# %% [markdown]
# ## Global variables

# %%
CACHE_DIR = '~/Data/huggignface/' # put your path here
RESULTS_FILE = 'Results/Baselines_v2/trec-6_results_.pkl' # put your path here

# %%


# %% [markdown]
# ## Dataset
print("CARGAR EL DATASET\n")

# %%
# rename correct column as 'labels': depends on the dataset you load

def tokenize(sample):
    """Tokenize sample"""
    
    sample = tokenizer(sample['text'], truncation=True, padding=False, return_length=True)
    
    return sample
    
def load_and_enrich_dataset(dataset_name, split, cache_dir):
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
# tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# train set
full_train_dataset = load_and_enrich_dataset('trec', split='train', cache_dir=CACHE_DIR).sort("lengths")

# mini train and val sets: 70%/30% of 30%
# XXX here we increase the train_size split since small dataset
train_val_datasets = full_train_dataset.train_test_split(train_size=0.99, shuffle=True)
train_val_datasets = train_val_datasets['train'].train_test_split(train_size=0.7, shuffle=True)
train_dataset = train_val_datasets['train'].sort("lengths")
val_dataset = train_val_datasets['test'].sort("lengths")

# test set
test_dataset = load_and_enrich_dataset('trec', split='test', cache_dir=CACHE_DIR).sort("lengths")

dataset_d = {
    'full_train': full_train_dataset,
    'train': train_dataset,
    'val': val_dataset,
    'test': test_dataset
    }

# dataloaders
dataloader_d = {}
for k, v in dataset_d.items():
    dataloader_d[k] = torch.utils.data.DataLoader(v, batch_size=256, collate_fn=DataCollatorWithPadding(tokenizer))

# %%
print("\nDATASET CARGADO:\n", dataset_d, "\n")


# %% [markdown]
# ## Optimization
print("OPTIMIZACIÓN", "\n")

# %%
baseline_params = {
                'embedding': 'bert-base-uncased', # TEXT.vocab.vectors,
                'distribution' : 'uniform',               # uniform, gaussian
                'input_dim' : 768,                        # dim of encoding!
                'reservoir_dim' : 0,      # not used
                'bias_scaling' : 0.0,     # not used
                'sparsity' : 0.0,         # not used
                'spectral_radius' : None, 
                'leaking_rate': 0.5,      # not used
                'activation_function' : 'tanh',
                'input_scaling' : 0.1,
                'mean' : 0.0,
                'std' : 1.0,
                'learning_algo' : None,
                'criterion' : None,
                'optimizer' : None,
                'pooling_strategy' : None,
                'lexicon' : None,
                'bidirectional' : False,
                'mode' : 'no_layer',     # simple baseline
                'device' : device,
                'seed' : 4
                }

# %%
results_d = {}

for pooling_strategy in tqdm(['last', 'mean']):
    
    results_d[pooling_strategy] = {}
    
    for alpha in tqdm([0.1, 1.0, 10.0, 100.0]):
        
        results_d[pooling_strategy][alpha] = []
        
        # model
        baseline_params['pooling_strategy'] = pooling_strategy
        baseline_params['mode'] = 'no_layer'
        print("\nPARAMS:\n",baseline_params)
        ESN = esn.EchoStateNetwork(**baseline_params)
        ESN.learning_algo = la.RidgeRegression(alpha=alpha)
        ESN = ESN.to(device)

        # train
        t0 = timer()
        LOSS = ESN.fit(dataloader_d["full_train"]) # full train set
        t1 = timer()
        # accuracy
        acc = ESN.predict(dataloader_d["test"], verbose=False)[1] # full test set

        if hasattr(acc, "item"):
            acc = acc.item()

        # results
        results_d[pooling_strategy][alpha].append([acc, t1 - t0])
        
        # clean objects
        del ESN.learning_algo
        del ESN.criterion
        del ESN.pooling_strategy
        del ESN
        torch.cuda.empty_cache()

# %%
print("RESULTS\n", results_d)


# %% [markdown]
# ## Results

# %%
# save results

with open(RESULTS_FILE, 'wb') as fh:
    pickle.dump(results_d, fh)

# %%
# # load results
# with open(os.path.join(RESULTS_PATH, RESULTS_FILE), 'rb') as fh:
#     results_d = pickle.load(fh)

# %%
# results_d

# %%

