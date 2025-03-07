# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import json
from sklearn.mixture import GaussianMixture
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import numpy as np


'''
提取来自llama2的隐藏表征
'''

# text_list = []
# with open('/home/ydk/dicken/LLaMA-Factory/data/HaluEval_Multitask_Hallucinat_Finally.json', 'r') as file:
#     data = json.load(file) 
#     for item in data:
#         instruction = item['instruction']
#         input = item['input']
#         vanilla_text = instruction + input
#         # print(vanilla_text)
#         text_list.append(vanilla_text)
#         # print(len(text_list))
# print("len(text_list)", len(text_list))
# batch_size = 512  
# num_batches = (len(text_list) + batch_size - 1) // batch_size  
# print(num_batches)
# all_hidden_states = torch.Tensor([])

# max_length = 128
# model_path = "/home/ydk/dicken/llm_models/Llama-2-7b-chat-hf"
# model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, output_hidden_states=True)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.pad_token = tokenizer.eos_token

# for i in range(num_batches):
    
#     batch_texts = text_list[i * batch_size: (i + 1) * batch_size]
  
#     inputs = tokenizer.batch_encode_plus(batch_texts, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
#     with torch.no_grad():   
#         outputs = model(**inputs)   
#     hidden_states = outputs.hidden_states
#     last_layer_hidden_states = hidden_states[-1]
        
#     all_hidden_states = torch.cat([all_hidden_states, last_layer_hidden_states], dim = 0)
#     print("batch_all_hidden_states", all_hidden_states.shape)
    
# print("all_hidden_states", all_hidden_states.shape)
# torch.save(all_hidden_states, '/home/ydk/dicken/LLaMA-Factory/checkpoint/all_hidden_states.pth')

'''
加载隐藏表征->pca降维->gmm聚类
'''
print("start loading the states!")
hidden_states = torch.load('/home/ydk/dicken/LLaMA-Factory/checkpoint/all_hidden_states.pth')  #torch.float32  .to(torch.float16)
# print(hidden_states.dtype)  
hidden_states = hidden_states.mean(dim = 1)
# vanilla = torch.randn(30000, 128, 4096).mean(dim = 1)
pca = PCA(n_components=512)
compressed_feature = pca.fit_transform(hidden_states)
print(compressed_feature.shape)
# x = torch.randn(1, 512)
n_components = [8, 16, 32, 64, 128]
for item in n_components:
    gmm = GaussianMixture(n_components=item, init_params="k-means++")
    gmm.fit(compressed_feature)
    # labels = gmm.predict(cluster_centers) #和batch中样本数量相同的labels，确定样本属于哪个类别
    cluster_centers = gmm.means_
    covariances = gmm.covariances_ #获取协方差矩阵
    # print("cluster_centers", cluster_centers.shape, cluster_centers.dtype) #float64
    # torch.save(cluster_centers, f'/home/ydk/dicken/LLaMA-Factory/checkpoint/new_cluster_centers_num{item}.pth')
    # covariances = gmm.covariances_ #获取协方差矩阵
    print("cluster_centers", cluster_centers.shape, cluster_centers.dtype, covariances.shape, covariances.dtype) #float64
    torch.save(cluster_centers, f'/home/ydk/dicken/LLaMA-Factory/checkpoint/cluster_centers_num{item}.pth')
    torch.save(covariances, f'/home/ydk/dicken/LLaMA-Factory/checkpoint/covariances_num{item}.pth')






