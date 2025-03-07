# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse
import pandas as pd
from rouge import Rouge
import bert_score
from bert_score import score
from bert_score import BERTScorer
from bert_score import plot_example
import ssl
import urllib.request
import zipfile
import sys
from pathlib import Path 


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

factual_eva = []
tokenizer_factkb = AutoTokenizer.from_pretrained("CDT/llm_models/FactKB", padding="max_length", truncation=True)
factkb = AutoModelForSequenceClassification.from_pretrained("CDT/llm_models/FactKB", num_labels=2)


base_path = "..."


file_paths = []
for i in range(7):  
    file_path = os.path.join(base_path, f"result_{i}_response.json")
    file_paths.append(file_path)

data = []
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
        for line in split_data:
            data.append(line)
            
batch_size = 256  
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    articles = [item["user_query"] for item in batch]  
    predictions = [item["prediction"] for item in batch]

    inputs = [(prediction, article) for prediction, article in zip(predictions, articles)]
    tokens = tokenizer_factkb(inputs, return_tensors="pt", padding="max_length", truncation=True)
    logits = factkb(**tokens).logits
    results = torch.softmax(logits, dim=1)
    result = results[:,1]

    for factual_score in result:
      
        factual_eva.append(factual_score.item())
average_score = sum(factual_eva) / len(factual_eva) * 100
print(average_score)

