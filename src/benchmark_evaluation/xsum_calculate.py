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



import json
file_path = '...'
shard_num = 8
Bert_P =  0
Bert_R = 0
Bert_F1 = 0
F1_rouge1 = 0
F1_rouge2 = 0
F1_rougeL = 0

for i in range(shard_num):
    fn = file_path + "_{:d}.json".format(i)
    with open(fn, "r") as f:
        content = json.load(f)
        Bert_P += content['Bert_P']
        Bert_R += content['Bert_R']
        Bert_F1 += content['Bert_F1']
        F1_rouge1 += content['F1_rouge1']
        F1_rouge2 += content['F1_rouge2']
        F1_rougeL += content['F1_rougeL']

print('Bert_P:', Bert_P / shard_num)
print('Bert_R:', Bert_R / shard_num)
print('Bert_F1:', Bert_F1 / shard_num)
print('F1_rouge1:', F1_rouge1 / shard_num)
print('F1_rouge2:', F1_rouge2 / shard_num)
print('F1_rougeL:', F1_rougeL / shard_num)
        


