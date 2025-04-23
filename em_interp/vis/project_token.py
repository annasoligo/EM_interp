'''
An interactive python file to pass a text through the model and collect its activations.
Then score each token activation based on the strength of its projection onto a loaded torch vector
(where the loaded vector is e.g. a DIM steering vector or log reg probe for a specific layer).
Texts are visualised with coloured text based on these token scores.
'''

# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import torch.nn as nn

from util.model_util import load_quantized_model