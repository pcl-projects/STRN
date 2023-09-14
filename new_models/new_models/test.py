#! /usr/bin/env python3


"""https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f"""


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

t = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(len(t.shape))
