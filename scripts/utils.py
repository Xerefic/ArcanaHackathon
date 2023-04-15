import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import openai
openai.organization = OPENAI_ORGANIZATION
openai.api_key = OPENAI_API_KEY

def get_embedding(text, net="text-embedding-ada-002"):
    res = openai.Embedding.create(
        input=[text], engine=net
    )
    return res['data'][0]['embedding']

def compute_historic_volatility(series, window):
    if window==-1:
        window = len(series['log_returns'])
    historic_volatility = np.sqrt(252) * series['log_returns'].rolling(window=window).std()
    return historic_volatility.dropna()

def compute_value_at_risk(series, window, confidence=0.95):
    if window==-1:
        window = len(series['log_returns'])
    value_at_risk = -series['log_returns'].rolling(window=window).quantile(1-confidence)
    return value_at_risk.dropna()