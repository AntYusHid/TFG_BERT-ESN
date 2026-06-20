# %% [markdown]
# # TREC-6
# # ESN Bayesian Optimization (``ax-platform``) + Simulations

# %% [markdown]
# ## Librairies

# %%
# !pip install transformers==4.8.2
# !pip install datasets==1.7.0
# !pip install ax-platform==0.1.20

# %%
print("INICIO DE LA EJECUCIÓN")
import os
import sys
sys.path.insert(0, os.path.abspath("../..")) # comment this if library is pip installed

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
print("IMPORTS FINALIZADOS")

# %%
# %config Completer.use_jedi = False
# %load_ext autoreload
# %autoreload 2

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%
SEED = 42

# %%


# %% [markdown]
# ## Global variables

# %%
pooling_strategy = 'mean' # 'mean' or 'last'

# %%
RESULTS_PATH = 'Results/ESN_v2' # put your path here
CACHE_DIR = '/huggingface_datasets' # put your path here
PARAMS_FILE = 'trec-6_params_uniform_'+pooling_strategy+'-pooling.pkl'
RESULTS_FILE = 'trec-6_results_esn_uniform_'+pooling_strategy+'-pooling_FINAL.pkl' # 

# %%
RESULTS_FILE

# %%
print("VARIABLES GLOBALES CARGADAS")

# %% [markdown]
# ## Dataset
print("CARGAR EL DATASET")

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
    print("Load dataset:\n", dataset, "\n")
    
    # Rename label column for tokenization purposes (use 'label-fine' for fine-grained labels)
    dataset = dataset.rename_column('label-coarse', 'labels')
    print("Rename label column:\n", dataset, "\n")
    
    # Tokenize data
    dataset = dataset.map(tokenize, batched=True)
    print("Tokenize data:\n", dataset, "\n")
    dataset = dataset.rename_column('length', 'lengths')
    print("Rename length column:\n", dataset, "\n")
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'lengths'])
    print("Dataset set format:\n", dataset, "\n", "Aplica el tipo 'torch' a las columnas añadidas en la tokenización\n")

    print()
    
    return dataset

# %%
# tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print("tokenizer:\n", tokenizer, "\n")

# train set
# full_train_dataset = load_and_enrich_dataset('trec', split='train', cache_dir=CACHE_DIR).sort("lengths")

# mini train and val sets: 70%/30% of 30%
# XXX here we increase the train_size split since small dataset
# train_val_datasets = full_train_dataset.train_test_split(train_size=0.99, shuffle=True)
# train_val_datasets = train_val_datasets['train'].train_test_split(train_size=0.7, shuffle=True)
# train_dataset = train_val_datasets['train'].sort("lengths")
# val_dataset = train_val_datasets['test'].sort("lengths")

# test set
test_dataset = load_and_enrich_dataset('trec', split='test', cache_dir=CACHE_DIR).sort("lengths")

dataset_d = {
    # 'full_train': full_train_dataset,
    # 'train': train_dataset,
    # 'val': val_dataset,
    'test': test_dataset
    }

# dataloaders
dataloader_d = {}
for k, v in dataset_d.items():
    dataloader_d[k] = torch.utils.data.DataLoader(v, batch_size=256, collate_fn=DataCollatorWithPadding(tokenizer))

# %%
print("\ndataset_d:\n",dataset_d, "\n")
print(test_dataset)
print(test_dataset['text'][0])
print(test_dataset['input_ids'][0])
print(test_dataset['text'][1])
print(test_dataset['input_ids'][1])
print(test_dataset['text'][2])
print(test_dataset['input_ids'][2])
print(test_dataset['text'][3])
print(test_dataset['input_ids'][3])
print()

# %%


# %%
print("Computing BERT embeddings once...")

bert = BertModel.from_pretrained("bert-base-uncased").to(device)
print("\n",bert)
bert.eval()

def compute_full_embeddings(dataloader):
    """
    Devuelve:
      - all_embeddings: lista de tensores (seq_len_i, 768) SIN padding
      - all_labels:     tensor de enteros
      - all_lengths:    tensor de enteros
    """
    all_embeddings = [] # Lista de embeddings
    all_labels = [] # Lista de etiquetas
    all_lengths = [] # Lista de longitudes

    with torch.no_grad():
        for batch in tqdm(dataloader):
            labels   = batch["labels"]            # (B,)
            lengths  = batch["lengths"]           # (B,)

            inputs = {
                "input_ids":      batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            # Cálculo de los embeddings de BERT
            outputs = bert(**inputs)
            # El último estado de BERT son los embeddings
            hidden = outputs.last_hidden_state    # (B, T_pad, 768)

            # Recortamos el padding de cada muestra individualmente
            for i in range(hidden.size(0)):
                l = lengths[i].item()
                all_embeddings.append(hidden[i, :l, :].cpu())  # (l, 768)

            all_labels.append(labels)
            all_lengths.append(lengths)

            del outputs
            torch.cuda.empty_cache()

    return all_embeddings, torch.cat(all_labels), torch.cat(all_lengths)

# Precomputamos para cada split
# emb_full_train_X, emb_full_train_y, emb_full_train_l = compute_full_embeddings(dataloader_d["full_train"])
# emb_train_X,      emb_train_y,      emb_train_l      = compute_full_embeddings(dataloader_d["train"])
# emb_val_X,        emb_val_y,        emb_val_l        = compute_full_embeddings(dataloader_d["val"])
emb_test_X,       emb_test_y,       emb_test_l       = compute_full_embeddings(dataloader_d["test"])

# Liberamos BERT de memoria — ya no lo necesitamos
del bert
torch.cuda.empty_cache()

# Guardado opcional para no recomputar entre sesiones
torch.save(
    {
        # "full_train": (emb_full_train_X, emb_full_train_y, emb_full_train_l),
        # "train":      (emb_train_X,      emb_train_y,      emb_train_l),
        # "val":        (emb_val_X,        emb_val_y,        emb_val_l),
        "test":       (emb_test_X,       emb_test_y,       emb_test_l),
    },
    "trec_test_embeddings.pt"
)

print("\nEMBEDDINGS DE BERT CALCULADOS","\n")

# %%
from torch.utils.data import Dataset

print("DATASET DE EMBEDDINGS DE BERT","\n")

class EmbeddingDataset(Dataset):
    """Dataset de embeddings BERT precomputados."""
    def __init__(self, embeddings, labels, lengths):
        self.embeddings = embeddings  # lista de tensores (l_i, 768)
        self.labels     = labels      # tensor (N,)
        self.lengths    = lengths     # tensor (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],   # (lengths_i, 768) [500]
            "labels":     self.labels[idx],
            "lengths":    self.lengths[idx],
            "input_ids":  self.lengths[idx],      # ← dummy escalar; ESN.warm_up
                                                  #   hará unsqueeze(0) → shape [1]
        }


def embedding_collate_fn(batch):
    """Hace padding de los embeddings al máximo largo del batch."""
    embeddings = [item["embeddings"] for item in batch]
    labels     = torch.stack([item["labels"]  for item in batch])
    lengths    = torch.stack([item["lengths"] for item in batch])

    max_len = max(e.size(0) for e in embeddings)
    hidden_dim = embeddings[0].size(1)  # 768
    padded = torch.zeros(len(embeddings), max_len, hidden_dim)
    for i, e in enumerate(embeddings):
        padded[i, :e.size(0), :] = e

    return {
        "embeddings": padded,   # (B, T_max, 768)
        "labels":     labels,
        "lengths":    lengths,
        "input_ids":  torch.zeros(len(batch), 1, dtype=torch.long),  # [B, 1] dummy
    }


# Construimos los datasets y dataloaders de embeddings
emb_dataset_d = {
    # "full_train": EmbeddingDataset(emb_full_train_X, emb_full_train_y, emb_full_train_l),
    # "train":      EmbeddingDataset(emb_train_X,      emb_train_y,      emb_train_l),
    # "val":        EmbeddingDataset(emb_val_X,         emb_val_y,       emb_val_l),
    "test":       EmbeddingDataset(emb_test_X,        emb_test_y,      emb_test_l),
}

emb_dataloader_d = {
    k: torch.utils.data.DataLoader(
            v, batch_size=256, collate_fn=embedding_collate_fn
        )
    for k, v in emb_dataset_d.items()
}

# %%
class PrecomputedEmbedding(nn.Module):
    """
    Módulo que sustituye a BERT dentro del ESN.
    En lugar de calcular embeddings, simplemente extrae el tensor
    precomputado del batch y lo transpone al formato que espera _embed.
    """
    def forward(self, batch):
        # batch["embeddings"]: [Batch_Size, Length, Embedding_Size]  →  devuelve [Length, Batch_Size, Embedding_Size]
        return batch["embeddings"].permute(1, 0, 2)

print("Etiquetas:\n", emb_dataset_d['test'].labels)
print("\nTamaño de test:", emb_dataset_d['test'].labels.shape)
print()
print("Longitudes:\n", emb_dataset_d['test'].lengths)
print()
print(emb_dataset_d['test'].embeddings[0])
print()
print(emb_dataset_d['test'].embeddings[0].shape)
print()
print(emb_dataset_d['test'].embeddings[-1])
print()
print(emb_dataset_d['test'].embeddings[-1].shape)
print()

# %% [markdown]
# ## Optimization
print("OPTIMIZACIÓN", "\n")

# %% [markdown]
# - Parameters

# %%
ESN_PARAMS = {
            'embedding': None, #'bert-base-uncased', # TEXT.vocab.vectors,
            'distribution' : 'uniform',               # uniform, gaussian
            'input_dim' : 768,                        # dim of encoding!
            'reservoir_dim' : None,
            'bias_scaling' : 0.0,
            'sparsity' : None,
            'spectral_radius' : None,
            'leaking_rate': None,
            'activation_function' : 'tanh',
            'input_scaling' : None,
            'mean' : 0.0,
            'std' : 1.0,
            'learning_algo' : None,
            'criterion' : None,
            'optimizer' : None,
            'pooling_strategy' : pooling_strategy,
            'lexicon' : None,
            'bidirectional' : False, # False
            'mode' : None,
            'device' : device,
            'seed' : None
             }

# %%
def plug_parameters(args_d):
    
    params_d = ESN_PARAMS.copy()
    
    for k, v in args_d.items():
        if k in params_d.keys():
            params_d[k] = v

    # params_d['reservoir_dim'] = int(params_d['reservoir_dim'])
    # params_d['spectral_radius'] = float(params_d['spectral_radius'])
    # params_d['leaking_rate'] = float(params_d['leaking_rate'])
    # params_d['input_scaling'] = float(params_d['input_scaling'])
    # params_d['sparsity'] = float(params_d['sparsity'])

    params_d['dim'] = int(args_d['reservoir_dim'])
            
    return params_d

# %%
# args_d = {'reservoir_dim': 1000, 
#      'spectral_radius': 0.9, 
#      'leaking_rate': 0.5, 
#      'input_scaling': 0.1, 
#      'sparsity': 0.99, 
#      'activation_function': 'tanh', 
#      'alpha': 1.0}

# d = plug_parameters(args_d)

# d

# %% [markdown]
# - Fitness function

# %%
def warm_up(ESN, emb_dataset_d):
    """Warm up usando embeddings precomputados."""
    nb_sentences = 10
    subset = torch.utils.data.Subset(emb_dataset_d["test"], range(nb_sentences))
    ESN.warm_up(subset) # ESN.warm_up itera, hace unsqueeze y BatchEncoding internamente
    
    # """Warm up the ESN."""
    
    # nb_sentences = 10
    
    # for i in range(nb_sentences):
    #     # Seleccionamos una muestra del EmbeddingDataset
    #     sample = emb_dataset_d["train"][i]
    #     # warm_up espera un batch con las claves adecuadas
    #     batch = embedding_collate_fn([sample])
    #     ESN.warm_up([batch])

        # sentence = dataset_d["train"].select([i])
        # dataloader_tmp = torch.utils.data.DataLoader(sentence, 
        #                                              batch_size=1, 
        #                                              collate_fn=DataCollatorWithPadding(tokenizer))  

        # ESN.warm_up(sentence)


# %%
def fitness(reservoir_dim=1000,
            spectral_radius=0.9,
            leaking_rate=0.5,
            input_scaling=0.1, 
            sparsity=0.99,
            activation_function='tanh',
            alpha=1.0,
            emb_dataset_d=emb_dataset_d, # dataset_d, 
            emb_dataloader_d=emb_dataloader_d, # dataloader_d, 
            seed_l=[1234321], # 420, 888],# , 1979, 7], # 5 seeds
            return_test_acc=False):
    
    if return_test_acc:
        seed_l=[1654547824] #, 1947, 1949, 1979, 1083, 1234321, 7, 42, 888, 5998989775] # 10 seeds
    elif not return_test_acc:
        seed_l=[1234321] #, 420, 888] # 3 seeds
    print('\nSEEDS:', seed_l)
    
    # create esn parameters
    args_d = locals()
    esn_params = plug_parameters(args_d)
    print("\nPARÁMETROS ESN:")
    for k,v in esn_params.items():
        print(f'{k}: {v}')
    
    # BO mode: param optimization (esn_params['mode'] = 'recurrent_layer')
    if not return_test_acc:
        print("\nOPTIMIZACIÓN BAYESIANA:\n")

        acc_l = []
    
        for seed in tqdm(seed_l):
            
            # model
            esn_params['mode'] = 'recurrent_layer'
            esn_params['seed'] = seed
            ESN = esn.EchoStateNetwork(**esn_params)
            ESN.layer.embedding = PrecomputedEmbedding()
            ESN.learning_algo = la.RidgeRegression(alpha = alpha)
            ESN = ESN.to(device)

            # warm up
            print("Warm up\n")
            warm_up(ESN, emb_dataset_d)
            
            # train
            print("Train\n")
            LOSS = ESN.fit(emb_dataloader_d["test"])                           # mini train set
            acc = ESN.predict(emb_dataloader_d["test"], verbose=False)[1] # mini val set
            acc_l.append(acc)
            
            # clean objects
            del ESN.learning_algo
            del ESN.criterion
            del ESN.pooling_strategy
            del ESN
            torch.cuda.empty_cache()
            
        # results
        return np.mean(acc_l) 
          
    # prediction mode: once params have been optimized
    elif return_test_acc:
        print("\nMODO PREDICCIÓN:\n")
        print('return_test_acc mode')
        
        # esn
        acc_esn_l = []
        time_esn_l = []
        # custom baseline
        acc_bs_l = []
        time_bs_l = []
        
        for seed in tqdm(seed_l):
            
            for mode in tqdm(['recurrent_layer', 'linear_layer']):
            
                # model
                esn_params['mode'] = mode
                esn_params['seed'] = seed
                ESN = esn.EchoStateNetwork(**esn_params)
                ESN.layer.embedding = PrecomputedEmbedding()  # ← inyección post-construcción
                ESN.learning_algo = la.RidgeRegression(alpha = alpha)
                ESN = ESN.to(device)
                print(ESN.layer.dim)

                # warm up solo para la capa recurrente
                if mode == 'recurrent_layer':
                    warm_up(ESN, emb_dataset_d)

                # train
                t0 = timer()
                LOSS = ESN.fit(emb_dataloader_d["test"])                       # full train set
                t1 = timer()
                acc = ESN.predict(emb_dataloader_d["test"], verbose=False)[1] # full test set
                
                # clean objects
                del ESN.learning_algo
                del ESN.criterion
                del ESN.pooling_strategy
                del ESN
                torch.cuda.empty_cache()
                
                # results
                if mode == 'recurrent_layer':
                    time_esn_l.append(t1 - t0)
                    acc_esn_l.append(acc)
                elif mode == 'linear_layer':
                    time_bs_l.append(t1 - t0)
                    acc_bs_l.append(acc)
            
        results_d = {}
        results_d['recurrent_layer'] = [np.mean(acc_esn_l), np.std(acc_esn_l), np.mean(time_esn_l), np.std(time_esn_l)]
        results_d['linear_layer'] = [np.mean(acc_bs_l), np.std(acc_bs_l), np.mean(time_bs_l), np.std(time_bs_l)]
            
        return results_d

# %%
# %%time

# fitness(
#         reservoir_dim=1000,
#         spectral_radius=0.9,
#         leaking_rate=0.5,
#         input_scaling=0.1, 
#         sparsity=0.99,
#         activation_function='tanh',
#         alpha=1.0,
#         dataset_d=dataset_d, 
#         dataloader_d=dataloader_d,
#         seed_l=[1234321, 7],
#         return_test_acc=True
# )

# %%
def wrapped_fitness(d, return_test_acc=False):
    
    return fitness(reservoir_dim=d['reservoir_dim'], # will be in the loop 
                   spectral_radius=d['spectral_radius'],
                   leaking_rate=d['leaking_rate'],
                   input_scaling=d['input_scaling'],
                   sparsity=d['sparsity'],
                   activation_function=d['activation_function'],
                   alpha=d['alpha'],
                   emb_dataset_d=emb_dataset_d, # dataset_d,
                   emb_dataloader_d=emb_dataloader_d, # dataloader_d,
                   return_test_acc=return_test_acc)

# %%
# *** WARNING *** DO NO EXECUTE NEXT CELLS IF BIDIRECTIONAL MODE (OPTIM ALREADY DONE)

# %% [markdown]
# - Optimization

# %%
best_params_d = {}

for res_dim in tqdm([100]):
    
    best_parameters, best_values, experiment, model = optimize(
            parameters=[
              {
                "name": "reservoir_dim",
                "value_type": "int",
                "type": "fixed",
                "value": res_dim,
              },
              {
                "name": "activation_function",
                "value_type": "str",
                "type": "fixed",
                "value": 'tanh',
              },
              {
                "name": "spectral_radius",
                "value_type": "float",
                "type": "choice",
                "values": [0.5, 1.0, 1.5],
              },
              {
                "name": "leaking_rate",
                "value_type": "float",
                "type": "choice",
                "values": [0.1, 0.5, 0.9],
              },
              {
                "name": "input_scaling",
                "value_type": "float",
                "type": "choice",
                "values": [0.1, 1.0],
              },
              {
                "name": "sparsity",
                "value_type": "float",
                "type": "choice",
                "values": [0.0, 0.9],
              },
              {
                "name": "alpha",
                "value_type": "float",
                "type": "choice",
                "values": [0.1, 1.0, 10.0, 100.0],
              }
            ],
            # Booth function
            evaluation_function = wrapped_fitness,
            minimize = False,
            objective_name = 'val_accuracy',
            total_trials = 2 # nb iterations here 20 original
        )
    
    # results
    best_params_d[res_dim] = {}
    best_params_d[res_dim]['best_parameters'] = best_parameters
    best_params_d[res_dim]['best_values'] = best_values
    best_params_d[res_dim]['experiment'] = experiment
    # best_params_d[res_dim]['model'] = model

# %%
print("BEST PARAMS:\n", best_params_d)

# %%
# Crear la carpeta si no existe
os.makedirs(RESULTS_PATH, exist_ok=True)

# save parameters

with open(os.path.join(RESULTS_PATH, PARAMS_FILE), 'wb') as fh:
    pickle.dump(best_params_d, fh)

# %%
# load params

with open(os.path.join(RESULTS_PATH, PARAMS_FILE), 'rb') as fh:
    best_params_d = pickle.load(fh)
    
PARAMS_FILE

# %% [markdown]
# ## Simulations
print("SIMULACIÓN")
# - Using best paramameters

# %%
# results

results_d = {}

for res_dim in tqdm([100]):
    
    best_parameters = best_params_d[res_dim]['best_parameters']
    d = wrapped_fitness(best_parameters, return_test_acc=True)
    results_d[res_dim] = d
    print("Experiment finished.")

# %%
print("RESULTS:\n", results_d)

# %%
RESULTS_FILE

# %%
# save results

with open(os.path.join(RESULTS_PATH, RESULTS_FILE), 'wb') as fh:
    pickle.dump(results_d, fh)

# %%
# # load results
# with open(os.path.join(RESULTS_PATH, RESULTS_FILE), 'rb') as fh:
#     results_d = pickle.load(fh)

# %%


# %%



