"""Initialization for NMF"""

import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as sch
import warnings


def nnls(X, W):
    """执行非负最小二乘法（NNLS）以计算签名暴露度。
    
    参数:
    X : array-like of shape (n_features, n_samples)
        突变计数矩阵，其中行代表不同的突变类型，列代表不同的样本。
        
    W : array-like of shape (n_features, n_components)
        签名矩阵，其中行代表不同的突变类型，列代表不同的突变签名。
    
    返回:
    H : array-like of shape (n_components, n_samples)
        签名暴露度矩阵，其中行代表不同的签名，列代表不同的样本。
    """
    H = []  # 初始化一个空列表，用于存储每个样本的签名暴露度
    for x in X.T:  # 遍历X的每一列（即每个样本）
        h, _ = sp.optimize.nnls(W, x)  # 使用非负最小二乘法计算当前样本的签名暴露度
        H.append(h)  # 将计算得到的暴露度添加到列表H中
    H = np.array(H)  # 将列表H转换为NumPy数组
    H = H.T  # 转置数组H，使得其行代表签名，列代表样本
    return H  # 返回签名暴露度矩阵H

def initialize_nmf(X, n_components, init='cluster', init_normalize_W=None,
                   init_refit_H=None,
                   init_cluster_metric='cosine',
                   init_cluster_linkage='average',
                   init_cluster_max_ncluster=100, init_cluster_min_nsample=1):
    """
    非负矩阵分解（NMF）初始化算法。

    参数：
    X : array-like of shape (n_features, n_samples)
        待分解的输入矩阵。
    n_components : int
        分解的秩，即结果矩阵的列数。
    init : str, optional (默认='random')
        初始化算法：
            - 'cluster': 聚类算法。
    init_normalize_W : bool, optional
        是否对每个特征进行L1归一化。
    init_refit_H : bool, optional
        是否使用非负最小二乘法（NNLS）重新拟合H矩阵。
    init_cluster_metric : str, optional (默认='cosine')
        聚类时使用的度量方式。
    init_cluster_linkage : str, optional (默认='average')
        聚类时使用的链接方式。
    init_cluster_max_ncluster : int, optional (默认=100)
        最大聚类数。
    init_cluster_min_nsample : int, optional (默认=1)
        每个聚类中最少的样本数。

    返回：
    W : 二维数组
        初始化的W矩阵。
    H : 二维数组
        初始化的H矩阵。

    异常：
    ValueError
        如果初始化参数无效或自定义矩阵形状不正确。
    TypeError
        如果init_normalize_W或init_refit_H参数类型不正确。
    """

    # 确保输入矩阵X是浮点数类型的numpy数组
    if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
        X = np.array(X).astype(float)
    n_features, n_samples = X.shape

    # 聚类算法初始化
    W, H, _ = _init_cluster(X, n_components, metric=init_cluster_metric,
                            linkage=init_cluster_linkage,
                            max_ncluster=init_cluster_max_ncluster,
                            min_nsample=init_cluster_min_nsample)
    return W, H


def _init_cluster(X, n_components, metric='cosine', linkage='average',
                  max_ncluster=100, min_nsample=1):
    """
    初始化聚类算法的函数，用于非负矩阵分解。

    参数：
    X : 二维数组
        待分解的非负矩阵。
    n_components : 整数
        分解的秩，即结果矩阵的列数。
    metric : 字符串，可选（默认='cosine'）
        聚类时使用的度量方式。
    linkage : 字符串，可选（默认='average'）
        聚类时使用的链接方式。
    max_ncluster : 整数，可选（默认=100）
        最大聚类数。
    min_nsample : 整数，可选（默认=1）
        每个聚类中最少的样本数。

    返回：
    W : 二维数组
        初始化的W矩阵。
    H : 二维数组
        初始化的H矩阵。
    cluster_membership : 一维数组
        每个样本的聚类成员关系。

    异常：
    RuntimeError
        如果聚类初始化失败。

    """
    n_features, n_samples = X.shape  # 获取X的行数和列数
    XT_norm = normalize(X, norm='l1', axis=0).T  # 对X进行列归一化并转置
    d = sp.spatial.distance.pdist(XT_norm, metric=metric)  # 计算样本间的距离
    d = d.clip(0)  # 确保距离非负
    linkage = sch.linkage(d, method=linkage)  # 进行层次聚类
    for ncluster in range(n_components, np.min([n_samples, max_ncluster]) + 1):
        cluster_membership = sch.fcluster(linkage, ncluster, criterion='maxclust')  # 根据最大聚类数进行聚类
        if len(set(cluster_membership)) != ncluster:
            cluster_membership = sch.cut_tree(linkage, n_clusters=ncluster).flatten() + 1
            if len(set(cluster_membership)) != ncluster:
                warnings.warn('Number of clusters output by cut_tree or fcluster is not equal to the specified number of clusters',
                              UserWarning)
        W = []
        for i in range(1, ncluster + 1):
            if np.sum(cluster_membership == i) >= min_nsample:
                W.append(np.mean(XT_norm[cluster_membership == i, :], 0))  # 计算每个聚类的中心
        W = np.array(W).T
        if W.shape[1] == n_components:
            break
    if W.shape[1] != n_components:
        raise RuntimeError('Initialization with init=cluster failed.')  # 如果聚类初始化失败，抛出异常
    W = normalize(W, norm='l1', axis=0)  # 对W进行列归一化
    H = nnls(X, W)  # 使用非负最小二乘法计算H矩阵
    return W, H, cluster_membership  # 返回初始化的W和H矩阵以及聚类成员关系

def beta_divergence(A, B, beta=1, square_root=False):
    """Beta_divergence

    A and B must be float arrays.

    """
    if beta == 1 or beta == "kullback-leibler":
        # When B_ij = 0, KL divergence is not defined unless A_ij = 0.
        # Whenever A_ij = 0, then the contribution of the term A_ij log(A_ij/B_ij)
        # is considered as 0.
        A_data = A.ravel()
        B_data = B.ravel()
        indices = A_data > 0
        A_data = A_data[indices]
        B_data_remaining = B_data[~indices]
        B_data = B_data[indices]
        # Here we must take matrix additions first and then take sum.
        # Otherwise, the separate matrix sums will be too big and the small
        # differences will be lost, and we'll get 0.0 results.
        res = np.sum(A_data*np.log(A_data/B_data) - A_data + B_data)
        res = res + np.sum(B_data_remaining)
    elif beta == 2 or beta == 'frobenius':
        res = np.linalg.norm(A - B, ord=None) # 2-norm for vectors and frobenius norm for matrices
        res = res**2 / 2
    else:
        raise ValueError('Only beta = 1 and beta = 2 are implemented.')
    if square_root:
        res = np.sqrt(2*res)

    return res


def normalize_WH(W, H):
    normalization_factor = np.sum(W, 0)
    return W/normalization_factor, H*normalization_factor[:, None]