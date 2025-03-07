
import torch
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.cluster import MeanShift
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from torch.linalg import inv



print("start loading the states!")
hidden_states = torch.load('/home/ydk/dicken/LLaMA-Factory/checkpoint/cluster_centers_num32.pth') #torch.float32
# print(hidden_states.type)
if torch.is_tensor(hidden_states):
    print("1")
else: print("2")

# print(hidden_states.shape)
# # hidden_states = hidden_states.mean(dim = 1)
# # vanilla = torch.randn(30000, 128, 4096).mean(dim = 1)
# pca = PCA(n_components=48)
# compressed_feature = pca.fit_transform(hidden_states)
# print(compressed_feature.shape)
# # x = torch.randn(1, 512)
# n_components = [8, 16]
# for item in n_components:
#     gmm = GaussianMixture(n_components=item, init_params="k-means++")
#     gmm.fit(compressed_feature)
#     # labels = gmm.predict(cluster_centers) #和batch中样本数量相同的labels，确定样本属于哪个类别
#     cluster_centers = gmm.means_
#     covariances = gmm.covariances_ #获取协方差矩阵
#     print("cluster_centers", cluster_centers.shape, cluster_centers.dtype, covariances.shape) #float64
#     torch.save(cluster_centers, f'/home/ydk/dicken/LLaMA-Factory/checkpoint/new_cluster_centers_num{item}.pth')
#     torch.save(covariances, f'/home/ydk/dicken/LLaMA-Factory/checkpoint/covariances_num{item}.pth')




'''
更改的代码
'''
# to(dtype=torch.float16) 
cluster_centers = torch.from_numpy(torch.load("/home/ydk/dicken/LLaMA-Factory/checkpoint/new_cluster_centers_num16.pth")) #假设为(16, 48)
covariances = torch.from_numpy(torch.load("/home/ydk/dicken/LLaMA-Factory/checkpoint/covariances_num16.pth")) #假设为(16, 48, 48)
# print(cluster_centers.shape)
# cluster_centers = torch.randn(16, 4096)
new_sample = torch.randn(8, 128, 48)
new_sample = torch.from_numpy(torch.load("/home/ydk/dicken/LLaMA-Factory/checkpoint/new_cluster_centers_num8.pth"))
# new_sample = new_sample.mean(dim = 1)
corresponding_cluster = []
# print(new_sample[0:1, :].size())

for i in range(new_sample.size(0)):
    each_sample = new_sample[i:i+1, :]
    
    
    # 计算欧氏距离
    # print("each_sample", each_sample.shape)
    # distances = [
    #         mahalanobis(each_sample, cluster_centers[k], np.linalg.inv(covariances[k]))
    #         for k in range(len(cluster_centers))
    #     ]

    #     # 选择距离最小的质心
    # best_index = np.argmin(distances).cpu().detach().numpy()
    # print("best_index", best_index)
    # each_cluster = cluster_centers[best_index-1:best_index, :]
    # corresponding_cluster.append(each_cluster)
    
    Mahalanobis_distances = []
    for i in range(len(cluster_centers)):
        diff = (each_sample.squeeze(0)  - cluster_centers[i]).unsqueeze(1)
        # print("diff",diff.shape)
        inv_cov = inv(covariances[i])
        # print("inv_cov",inv_cov.shape)
        dist = torch.sqrt(torch.matmul(torch.matmul(diff.t(), inv_cov), diff))
        Mahalanobis_distances.append(dist.item())
    
    # 选择距离最小的质心
    closest_cluster_index = torch.argmin(torch.tensor(Mahalanobis_distances)).cpu().detach().numpy()
    print("closest_cluster_index", closest_cluster_index)
    each_cluster = cluster_centers[closest_cluster_index-1:closest_cluster_index, :]
    corresponding_cluster.append(each_cluster)
    
    
    
   
    
    
    # distances = torch.norm(cluster_centers - each_sample, dim=1)
    # # 将距离转换为概率
    # probabilities = torch.exp(-distances) / torch.sum(torch.exp(-distances))
    # # 确定新样本所属的簇
    # predicted_cluster = torch.argmax(probabilities).cpu().detach().numpy()
    # each_cluster = cluster_centers[predicted_cluster-1:predicted_cluster, :]
    # corresponding_cluster.append(each_cluster)

corresponding_cluster = torch.cat(corresponding_cluster, dim=0)
print("Predicted cluster:",corresponding_cluster.shape)
        
    
    # print(slice_tensor.size())



# # 计算新样本与每个簇中心的距离
# distances = np.linalg.norm(cluster_centers - new_sample, axis=1)  # 欧氏距离

# # 将距离转换为概率
# probabilities = np.exp(-distances) / np.sum(np.exp(-distances))

# # 确定新样本所属的簇
# predicted_cluster = np.argmax(probabilities)

# print("Predicted cluster:", predicted_cluster)
# print("Posterior probabilities:", probabilities)





