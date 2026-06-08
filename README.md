## Overview
This repository includes our works on Graph representation learning and its application on Traffic Flow Prediction.

The ideas behind our works can be abstracted and demonstrated in the following big picture. 

<p align = 'center'>
  <img width = "500" src= "./big picture.png">
</p>

All works can be deduced, inspired and created from this picture.

<p align = 'center'>
  <img src="./big picture2.png">
</p>

The congruent relationships between Our works included in this repository and the big picture are listed in the following:
<p align = 'center'>
  <img src="./Table.png">
</p>

The file structure is listed as follows:

1 T-GCN is the source codes for the paper named “T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction” published in IEEE Transactions on Intelligent Transportation Systems (T-ITS) which forged the T-GCN model the spatial and temporal dependence simultaneously.  

2 A3T-GCN is the source codes for the paper named “A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting” published at ISPRS International Journal of Geo-Information which strengthen the T-GCN model model with attention structure. 

3 AST-GCN is the source codes for the paper named “AST-GCN: Attribute-Augmented Spatiotemporal Graph Convolutional Network for Traffic Forecasting” published in IEEE Access which strengthen the T-GCN model model with attribute information. 

4 KST-GCN is the source codes for the paper named “KST-GCN: A Knowledge-Driven Spatial-Temporal Graph Convolutional Network for Traffic Forecasting” published in IEEE Transactions on Intelligent Transportation Systems (T-ITS) which  which strengthen the T-GCN model model with knowledge graph.

5 CGCN is the source code for the paper named “Curvature graph neural network” published at Information Sciences which used Ricci curvature information to model pivotal nodes. STCGNN is the source code for the paper named "Ollivier–Ricci Curvature Based Spatio-Temporal Graph Neural Networks for Traffic Flow Forecasting" which is an extension of CGCN in the field of traffic forecasting. 

6 STGC-GNNs is the source codes for the paper named "STGC-GNNs: A GNN-based traffic prediction framework with a spatial-temporal Granger causality graph" published in Physica A: Statistical Mechanics and its Applications. 

7 iGCL is the source codes for the paper named "Augmentation-Free Graph Contrastive Learning of Invariant-Discriminative Representations" published in IEEE Transactions on Neural Networks and Learning Systems. 

8. High-Order Topology-Enhanced Graph Convolutional Networks (HoT-GCN) for Dynamic Graphs. 

9. Alleviating neighbor bias: augmenting graph self-supervise learning with structural equivalent positive samples.  

10. LSTTN: A Long-Short Term Transformer-based Spatiotemporal Neural Network for Traffic Flow Forecasting published in Knowlege-based System.  

11. CAT: A Causally Graph Attention Network for Trimming Heterophilic Graph published in Information Science.
    
12. Causal invariant geographic network representations with feature and structural distribution shifts.

13. STDCformer: A transformer-based model with a spatial-temporal causal de-confounding strategy for crowd flow prediction

13 Baseline includes methods such as (1) History Average model (HA) (2) Autoregressive Integrated Moving Average model (ARIMA) (3) Support Vector Regression model (SVR) (4) Graph Convolutional Network model (GCN) (5) Gated Recurrent Unit model (GRU)

## 1. T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction
Accurate and real-time traffic forecasting plays an important role in the Intelligent Traffic System and is of great significance for urban traffic planning, traffic management, and traffic control. However, traffic forecasting has always been considered an open scientific issue, owing to the constraints of urban road network topological structure and the law of dynamic change with time, namely, spatial dependence and temporal dependence. To capture the spatial and temporal dependence simultaneously, we propose a novel neural network-based traffic forecasting method, the temporal graph convolutional network (T-GCN) model, which is in combination with the graph convolutional network (GCN) and gated recurrent unit (GRU). Specifically, the GCN is used to learn complex topological structures to capture spatial dependence and the gated recurrent unit is used to learn dynamic changes of traffic data to capture temporal dependence. Then, the T-GCN model is employed to traffic forecasting based on the urban road network. Experiments demonstrate that our T-GCN model can obtain the spatio-temporal correlation from traffic data and the predictions outperform state-of-art baselines on real-world traffic datasets.

The manuscript can be visited at https://ieeexplore.ieee.org/document/8809901 or https://arxiv.org/abs/1811.05320

[The code](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN)

## 2. A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting
Accurate real-time traffic forecasting is a core technological problem against the implementation of the intelligent transportation system. However, it remains challenging considering the complex spatial and temporal dependencies among traffic flows. In the spatial dimension, due to the connectivity of the road network, the traffic flows between linked roads are closely related. In terms of the temporal factor, although there exists a tendency among adjacent time points in general, the importance of distant past points is not necessarily smaller than that of recent past points since traffic flows are also affected by external factors. In this study, an attention temporal graph convolutional network (A3T-GCN) traffic forecasting method was proposed to simultaneously capture global temporal dynamics and spatial correlations. The A3T-GCN model learns the short-time trend in time series by using the gated recurrent units and learns the spatial dependence based on the topology of the road network through the graph convolutional network. Moreover, the attention mechanism was introduced to adjust the importance of different time points and assemble global temporal information to improve prediction accuracy. Experimental results in real-world datasets demonstrate the effectiveness and robustness of proposed A3T-GCN.

The manuscript can be visited at https://www.mdpi.com/2220-9964/10/7/485/html or arxiv https://arxiv.org/abs/2006.11583.

[The code](https://github.com/lehaifeng/T-GCN/tree/master/A3T-GCN)

## 3. AST-GCN: Attribute-Augmented Spatiotemporal Graph Convolutional Network for Traffic Forecasting
Traffic forecasting is a fundamental and challenging task in the field of intelligent transportation. Accurate forecasting not only depends on the historical traffic flow information but also needs to consider the influence of a variety of external factors, such as weather conditions and surrounding POI distribution. Recently, spatiotemporal models integrating graph convolutional networks and recurrent neural networks have become traffic forecasting research hotspots and have made significant progress. However, few works integrate external factors. Therefore, based on the assumption that introducing external factors can enhance the spatiotemporal accuracy in predicting traffic and improving interpretability, we propose an attribute-augmented spatiotemporal graph convolutional network (AST-GCN). We model the external factors as dynamic attributes and static attributes and design an attribute-augmented unit to encode and integrate those factors into the spatiotemporal graph convolution model. Experiments on real datasets show the effectiveness of considering external information on traffic speed forecasting tasks when compared with traditional traffic prediction methods. Moreover, under different attribute-augmented schemes and prediction horizon settings, the forecasting accuracy of the AST-GCN is higher than that of the baselines.

The manuscript can be visited at https://ieeexplore.ieee.org/document/9363197 or https://arxiv.org/abs/2011.11004.

[The code](https://github.com/lehaifeng/T-GCN/tree/master/AST-GCN)

## 4. KST-GCN: A Knowledge-Driven Spatial-Temporal Graph Convolutional Network for Traffic Forecasting
While considering the spatial and temporal features of traffic, capturing the impacts of various external factors on travel is an essential step towards achieving accurate traffic forecasting. However, existing studies seldom consider external factors or neglect the effect of the complex correlations among external factors on traffic. Intuitively, knowledge graphs can naturally describe these correlations. Since knowledge graphs and traffic networks are essentially heterogeneous networks, it is challenging to integrate the information in both networks. On this background, this study presents a knowledge representation-driven traffic forecasting method based on spatial-temporal graph convolutional networks. We first construct a knowledge graph for traffic forecasting and derive knowledge representations by a knowledge representation learning method named KR-EAR. Then, we propose the Knowledge Fusion Cell (KF-Cell) to combine the knowledge and traffic features as the input of a spatial-temporal graph convolutional backbone network. Experimental results on the real-world dataset show that our strategy enhances the forecasting performances of backbones at various prediction horizons. The ablation and perturbation analysis further verify the effectiveness and robustness of the proposed method. To the best of our knowledge, this is the first study that constructs and utilizes a knowledge graph to facilitate traffic forecasting; it also offers a promising direction to integrate external information and spatial-temporal information for traffic forecasting.

The manuscript can be visited at https://ieeexplore.ieee.org/document/9681326/ or https://arxiv.org/abs/2011.14992.

[The code](https://github.com/lehaifeng/T-GCN/tree/master/KST-GCN)

## 5. Curvature graph neural network
Graph neural networks (GNNs) have achieved great success in many graph-based tasks. Much work is dedicated to empowering GNNs with adaptive locality ability, which enables the measurement of the importance of neighboring nodes to the target node by a node-specific mechanism. However, the current node-specific mechanisms are deficient in distinguishing the importance of nodes in the topology structure. We believe that the structural importance of neighboring nodes is closely related to their importance in aggregation. In this paper, we introduce discrete graph curvature (the Ricci curvature) to quantify the strength of the structural connection of pairwise nodes. We propose a curvature graph neural network (CGNN), which effectively improves the adaptive locality ability of GNNs by leveraging the structural properties of graph curvature. To improve the adaptability of curvature on various datasets, we explicitly transform curvature into the weights of neighboring nodes by the necessary negative curvature processing module and curvature normalization module. Then, we conduct numerous experiments on various synthetic and real-world datasets. The experimental results on synthetic datasets show that CGNN effectively exploits the topology structure information and that the performance is significantly improved. CGNN outperforms the baselines on 5 dense node classification benchmark datasets. This study provides a deepened understanding of how to utilize advanced topology information and assign the importance of neighboring nodes from the perspective of graph curvature and encourages bridging the gap between graph theory and neural networks. The source code is available at https://github.com/GeoX-Lab/CGNN.

The manuscript can be visited at https://www.sciencedirect.com/science/article/pii/S0020025521012986 or https://arxiv.org/abs/2106.15762.

The extension of CGNN in the filed of traffic prediction is Spatio-temporal Cruvature Graph Neural Network(STCGNN). The manuscript can be visited at https://www.mdpi.com/2073-8994/15/5/995. 

[The code](https://github.com/GeoX-Lab/STCGNN)

## 6.STGC-GNNs: A GNN-based traffic prediction framework with a spatial-temporal Granger causality graph
It is important to model the spatial dependence of the road network for traffic prediction tasks. The essence of spatial dependence is to accurately describe how traffic information transmission is affected by other nodes in the road network, and the GNN-based traffic prediction model, as a benchmark for traffic prediction, has become the most common method for the ability to model spatial dependence by transmitting traffic information with the message passing mechanism. However, the transmission of traffic information is a global and dynamic process in long-term traffic prediction, which cannot be described by the local and static spatial dependence. In this paper, we proposed a spatial-temporal Granger causality(STGC) to model the global and dynamic spatial dependence, which can capture a stable causal relationship between nodes underlying dynamic traffic flow. The STGC can be detected by a spatial-temporal Granger causality test methods proposed by us. We chose T-GCN, STGCN and Graph Wavenet as bakbones, and the experimental results on three backbone models show that using STGC to model the spatial dependence has better results than the original model for 45-min and 1 h long-term prediction. 

The manuscript can be visited at https://www.sciencedirect.com/science/article/abs/pii/S0378437123004685 or https://arxiv.org/abs/2210.16789.

[The code](https://github.com/lehaifeng/T-GCN/tree/master/STGC-GNN)

## 7. Augmentation-Free Graph Contrastive Learning of Invariant-Discriminative Representations
Graph contrastive learning is a promising direction toward alleviating the label dependence, poor generalization and weak robustness of graph neural networks, learning representations with invariance, and discriminability by solving pretasks. The pretasks are mainly built on mutual information estimation, which requires data augmentation to construct positive samples with similar semantics to learn invariant signals and negative samples with dissimilar semantics in order to empower representation discriminability. However, an appropriate data augmentation configuration depends heavily on lots of empirical trials such as choosing the compositions of data augmentation techniques and the corresponding hyperparameter settings. We propose an augmentation-free graph contrastive learning method, invariant-discriminative graph contrastive learning (iGCL), that does not intrinsically require negative samples. iGCL designs the invariant-discriminative loss (ID loss) to learn invariant and discriminative representations. On the one hand, ID loss learns invariant signals by directly minimizing the mean square error between the target samples and positive samples in the representation space. On the other hand, ID loss ensures that the representations are discriminative by an orthonormal constraint forcing the different dimensions of representations to be independent of each other. This prevents representations from collapsing to a point or subspace. Our theoretical analysis explains the effectiveness of ID loss from the perspectives of the redundancy reduction criterion, canonical correlation analysis, and information bottleneck principle. The experimental results demonstrate that iGCL outperforms all baselines on 5 node classification benchmark datasets. iGCL also shows superior performance for different label ratios and is capable of resisting graph attacks, which indicates that iGCL has excellent generalization and robustness. 

The manuscript can be visited at arxiv https://arxiv.org/abs/2210.08345 or https://ieeexplore.ieee.org/document/10058898.

[The code](https://github.com/lehaifeng/T-GCN/tree/master/iGCL)

## 8. High-Order Topology-Enhanced Graph Convolutional Networks (HoT-GCN) for Dynamic Graphs
Understanding the evolutionary mechanisms of dynamic graphs is crucial since dynamic is a basic characteristic of real-world networks. The challenges of modeling dynamic graphs are as follows: (1) Real-world dynamics are frequently characterized by group effects, which essentially emerge from high-order interactions involving groups of entities. Therefore, the pairwise interactions revealed by the edges of graphs are insufficient to describe complex systems. (2) The graph data obtained from real systems are often noisy, and the spurious edges can interfere with the stability and efficiency of models. To address these issues, we propose a high-order topology-enhanced graph convolutional network for modeling dynamic graphs. The rationale behind it is that the symmetric substructure in a graph, called the maximal clique, can reflect group impacts from high-order interactions on the one hand, while not being readily disturbed by spurious links on the other hand. Then, we utilize two independent branches to model the distinct influence mechanisms of the two effects. Learnable parameters are used to tune the relative importance of the two effects during the process. We conduct link predictions on real-world datasets, including one social network and two citation networks. Results show that the average improvements of the high-order enhanced methods are 68%, 15%, and 280% over the corresponding backbones across datasets. The ablation study and perturbation analysis validate the effectiveness and robustness of the proposed method. Our research reveals that high-order structures provide new perspectives for studying the dynamics of graphs and highlight the necessity of employing higher-order topologies in the future.

The manuscript can be visited at https://www.mdpi.com/2073-8994/14/10/2218.

[The code](https://github.com/lehaifeng/T-GCN/tree/master/HoT-GCN)

## 9. Alleviating neighbor bias: augmenting graph self-supervise learning with structural equivalent positive samples
In recent years, using a self-supervised learning framework to learn the general characteristics of graphs has been considered a promising paradigm for graph representation learning. The core of self-supervised learning strategies for graph neural networks lies in constructing suitable positive sample selection strategies. However, existing GNNs typically aggregate information from neighboring nodes to update node representations, leading to an over-reliance on neighboring positive samples, i.e., homophilous samples; while ignoring long-range positive samples, i.e., positive samples that are far apart on the graph but structurally equivalent samples, a problem we call "neighbor bias." This neighbor bias can reduce the generalization performance of GNNs. In this paper, we argue that the generalization properties of GNNs should be determined by combining homogeneous samples and structurally equivalent samples, which we call the "GC combination hypothesis." Therefore, we propose a topological signal-driven self-supervised method. It uses a topological information-guided structural equivalence sampling strategy. First, we extract multiscale topological features using persistent homology. Then we compute the structural equivalence of node pairs based on their topological features. In particular, we design a topological loss function to pull in non-neighboring node pairs with high structural equivalence in the representation space to alleviate neighbor bias. Finally, we use the joint training mechanism to adjust the effect of structural equivalence on the model to fit datasets with different characteristics. We conducted experiments on the node classification task across seven graph datasets. The results show that the model performance can be effectively improved using a strategy of topological signal enhancement.

The manuscript can be visited at arxiv https://arxiv.org/abs/2212.04365.

[The code](https://github.com/lehaifeng/T-GCN/tree/master/Sep-GCL)

## 10. LSTTN: A Long-Short Term Transformer-based Spatiotemporal Neural Network for Traffic Flow Forecasting
Accurate traffic forecasting is a fundamental problem in intelligent transportation systems and learning long-range traffic representations with key information through spatiotemporal graph neural networks (STGNNs) is a basic assumption of current traffic flow prediction models. However, due to structural limitations, existing STGNNs can only utilize short-range traffic flow data; therefore, the models cannot adequately learn the complex trends and periodic features in traffic flow. Besides, it is challenging to extract the key temporal information from the long historical traffic series and obtain a compact representation. To solve the above problems, we propose a novel LSTTN (Long-Short Term Transformer-based Network) framework comprehensively considering the long- and short-term features in historical traffic flow. First, we employ a masked subseries Transformer to infer the content of masked subseries from a small portion of unmasked subseries and their temporal context in a pretraining manner, forcing the model to efficiently learn compressed and contextual subseries temporal representations from long historical series. Then, based on the learned representations, long-term trend is extracted by using stacked 1D dilated convolution layers, and periodic features are extracted by dynamic graph convolution layers. For the difficulties in making time-step level predictions, LSTTN adopts a short-term trend extractor to learn fine-grained short-term temporal features. Finally, LSTTN fuses the long-term trend, periodic features and short-term features to obtain the prediction results. Experiments on four real-world datasets show that in 60-minute-ahead long-term forecasting, the LSTTN model achieves a minimum improvement of 5.63% and a maximum improvement of 16.78% over baseline models.

The manuscript can be visited at [arXiv](https://arxiv.org/abs/2403.16495) or [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0950705124002727)

[The code](https://github.com/GeoX-Lab/LSTTN)

## 11. CAT: A Causally Graph Attention Network for Trimming Heterophilic Graph
The local attention-guided message passing mechanism (LAMP) adopted in graph attention networks (GATs) can adaptively learn the importance of neighboring nodes and perform local aggregation better, thus demonstrating a stronger discrimination ability. However, existing GATs suffer from significant discrimination ability degradations in heterophilic graphs. The reason is that a high proportion of dissimilar neighbors can weaken the self-attention of the central node, resulting in the central node deviating from its similar nodes in the representation space. This type of influence caused by neighboring nodes is referred to as Distraction Effect (DE) in this paper. To estimate and weaken the DE induced by neighboring nodes, we propose a Causal graph Attention network for Trimming heterophilic graphs (CAT). To estimate the DE, since DE is generated through two paths, we adopt the total effect as the metric for estimating DE; To weaken the DE, we identify the neighbors with the highest DE (we call them Distraction Neighbors) and remove them. We adopt three representative GATs as the base model within the proposed CAT framework and conduct experiments on seven heterophilic datasets of three different sizes. Comparative experiments show that CAT can improve the node classification accuracies of all base GAT models. Ablation experiments and visualization further validate the enhanced discrimination ability of CATs. In addition, CAT is a plug-and-play framework and can be introduced to any LAMP-driven GAT because it learns a trimmed graph in the attention-learning stage, instead of modifying the model architecture or globally searching for new neighbors.

The manuscript can be visited at [arXiv](https://arxiv.org/abs/2312.08672) or [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0020025524008302)

[The code](https://github.com/GeoX-Lab/CAT).

## 12. Causal invariant geographic network representations with feature and structural distribution shifts

Relationships between geographic entities, including human-land and human-people relationships, can be naturally modelled by graph structures, and geographic network representation is an important theoretical issue. The existing methods learn geographic network representations through deep graph neural networks (GNNs) based on the i.i.d. assumption. However, the spatial heterogeneity and temporal dynamics of geographic data make the out-of-distribution (OOD) generalisation problem particularly salient. We classify geographic network representations into invariant representations that always stabilise the predicted labels under distribution shifts and background representations that vary with different distributions. The latter are particularly sensitive to distribution shifts (feature and structural shifts) between testing and training data and are the main causes of the out-of-distribution generalisation (OOD) problem. Spurious correlations are present between invariant and background representations due to selection biases/environmental effects, resulting in the model extremes being more likely to learn background representations. The existing approaches focus on background representation changes that are determined by shifts in the feature distributions of nodes in the training and test data while ignoring changes in the proportional distributions of heterogeneous and homogeneous neighbour nodes, which we refer to as structural distribution shifts. We propose a feature-structure mixed invariant representation learning (FSM-IRL) model that accounts for both feature distribution shifts and structural distribution shifts. To address structural distribution shifts, we introduce a sampling method based on causal attention, encouraging the model to identify nodes possessing strong causal relationships with labels or nodes that are more similar to the target node. This approach significantly enhances the invariance of the representations between the source and target domains while reducing the dependence on background representations that arise by chance or in specific patterns. Inspired by the Hilbert–Schmidt independence criterion, we implement a reweighting strategy to maximise the orthogonality of the node representations, thereby mitigating the spurious correlations among the node representations and suppressing the learning of background representations. In addition, we construct an educational-level geographic network dataset under out-of-distribution (OOD) conditions. Our experiments demonstrate that FSM-IRL exhibits strong learning capabilities on both geographic and social network datasets in OOD scenarios.

The manuscript can be visited at [arXiv](https://arxiv.org/abs/2503.19382) or [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0167739X25001098)

[The code](https://github.com/GeoX-Lab/FSM-IRL).

## 13. STDCformer: A transformer-based model with a spatial-temporal causal de-confounding strategy for crowd flow prediction

 Crowd Flow Prediction is critical to urban management, with the goal of capturing the arrival and departure characteristics of crowd movements under different spatial and temporal distributions, which is fundamentally a spatial-temporal prediction task. Existing works typically treat spatial-temporal prediction as the task of learning a function F to transform historical observations to future observations. We further decompose this cross-time transformation into three processes: (1) Encoding (E): learning the in trinsic representation of observations, (2) Cross-Time Mapping (M): transforming past representations into future representations, and (3) Decoding (D): reconstructing future observations from the future representations. From this perspective, spatial-temporal prediction can be viewed as learning F = E · M·D, which includes learning the space transformations {E,D} between the observation space and the hidden representation space, as well as the spatial-temporal mapping M from future states to past states within
 the representation space. This leads to two key questions: Q1: What kind of representation space allows for mapping the past to the future? Q2:How to achieve mapping the past to the future within the representation space? To address Q1, we propose a Spatial-Temporal Backdoor Adjustment strategy, which learns a Spatial-Temporal De-Confounded (STDC) representation space and estimates the de-confounding causal effect of historical data on future data. This causal relationship we captured serves as the foundation for subsequent spatial-temporal mapping. To address Q2, we design a Spatial-Temporal Embedding (STE) that fuses the information of temporal and spatial confounders, capturing the intrinsic spatial-temporal characteristics of the representations. Additionally, we introduce a Cross-Time Attention mechanism, which queries the attention between the future and the past to guide spatial-temporal mapping. Finally, we integrate the process of learning the STDC representation space and the spatial-temporal mapping into an E-M-D skeleton for spatial-temporal prediction. The skeleton is further instantiated with a Transformer model, building a Transformer model with Spatial-Temporal De-Confounding Strategy (STDCformer). Experiments on two real-world datasets demonstrate that STDCformer achieves state-of-the-art predictive performance and exhibits stronger out-of-distribution generalization capabilities.


















下面是我联网检索后，对 **MPC 路径规划**相关论文做的理论梳理和时间线总结。先给一句总判断：**MPC 在路径规划里最核心的价值，不是“找一条几何最短路”，而是在运动学/动力学约束、控制输入约束、安全距离约束、动态障碍预测约束下，实时滚动求解一段未来最优轨迹。**

**一、MPC 是什么**

MPC，全称 Model Predictive Control，中文通常叫 **模型预测控制**。它的基本思想是：

在当前时刻，根据系统模型预测未来一段时间内的状态变化，然后求解一个有限时域优化问题，得到一串未来控制量；但真正执行时，只执行第一个控制量。下一时刻再重新感知环境、重新预测、重新优化。

这个过程也叫 **滚动时域控制** 或 **receding horizon control**。Mayne 等人在经典综述中把 MPC 概括为：每个采样时刻求解一个有限时域开环最优控制问题，然后只应用最优控制序列的第一个控制量。这个定义基本奠定了 MPC 理论框架。([colab.ws](https://colab.ws/articles/10.1016%2FS0005-1098%2899%2900214-9))

用路径规划语言说就是：

1. 当前机器人/AUV/车辆在位置 `x_t`。
2. 预测未来 `N` 步的状态：`x_{t+1}, x_{t+2}, ..., x_{t+N}`。
3. 同时优化未来 `N` 步的控制：`u_t, u_{t+1}, ..., u_{t+N-1}`。
4. 目标函数让轨迹尽量靠近目标、路径平滑、控制能耗小、远离障碍物。
5. 约束条件保证速度、加速度、转角、碰撞距离、动力学模型都合法。
6. 执行 `u_t`，下一步重新来一遍。

所以 MPC 的本质是：**把控制问题变成一个实时反复求解的约束优化问题。**

**二、MPC 的标准数学形式**

离散系统一般写成：

```text
x_{k+1} = f(x_k, u_k)
```

其中：

- `x_k` 是状态，比如位置、速度、航向角、角速度。
- `u_k` 是控制量，比如加速度、转角、推力、舵角、螺旋桨转速。
- `f` 是系统模型，可以是线性模型，也可以是非线性动力学模型。

MPC 在每个时刻求解：

```text
min Σ [状态误差代价 + 控制代价 + 控制变化代价 + 避障代价] + 终端代价
```

更具体地可以写成：

```text
min Σ_{i=0}^{N-1} [
  ||x_{k+i} - x_ref||_Q^2
  + ||u_{k+i}||_R^2
  + ||Δu_{k+i}||_S^2
  + obstacle_cost(x_{k+i})
]
+ ||x_{k+N} - x_goal||_P^2
```

约束包括：

```text
x_{k+i+1} = f(x_{k+i}, u_{k+i})
x_min ≤ x_{k+i} ≤ x_max
u_min ≤ u_{k+i} ≤ u_max
distance(x_{k+i}, obstacle) ≥ safe_margin
x_{k+N} ∈ terminal_set
```

这些项对应到路径规划里分别是：

- **状态误差代价**：希望靠近参考路径或目标点。
- **控制代价**：希望少用力、少耗能。
- **控制变化代价**：希望动作平滑，不突然急转或急刹。
- **避障代价/约束**：希望远离障碍物。
- **终端代价/终端集合**：保证预测时域末端不会走到无法继续控制的危险状态。

这也是为什么 MPC 很适合自动驾驶、无人机、AUV 和移动机器人：它可以天然处理多变量、多约束和未来预测。自动驾驶路径跟踪综述也强调，MPC 被广泛用于轨迹跟踪，正是因为它能系统处理状态约束、控制约束以及未来行为预测。([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S1367578822001377))

**三、从路径规划角度理解 MPC**

传统路径规划方法，例如 A*、Dijkstra、RRT、人工势场，很多时候偏向“几何路径”：找一条从起点到终点不碰障碍物的曲线。

MPC 更偏向“可执行轨迹”：不仅要不碰障碍，还要问这条轨迹能不能被真实系统开出来。

比如 AUV 或无人车不能瞬间横移，也不能无限加速，更不能突然从 0 度航向跳到 90 度航向。MPC 会把这些运动学和动力学限制放进优化问题中，因此它生成的不是抽象路径，而是带有速度、加速度、航向、控制输入的轨迹。

所以可以这样区分：

| 方法 | 主要关心 | 输出 |
|---|---|---|
| A* / Dijkstra | 栅格或图上的最短路径 | 几何路径 |
| RRT / RRT* | 高维空间可行路径搜索 | 采样路径 |
| 人工势场 | 目标吸引 + 障碍排斥 | 局部运动方向 |
| MPC | 模型约束下的未来最优控制 | 可执行轨迹 + 控制量 |

在实际系统中，常见结构是：

```text
全局规划器：A* / RRT* / PRM / 栅格地图
        ↓
局部规划器：MPC / NMPC / Tube-MPC
        ↓
底层控制器：PID / LQR / 推力分配 / 舵角控制
```

也有一些新论文会把 MPC 同时作为“局部路径规划 + 跟踪控制”模块，不再严格分开规划和控制。自动地面车辆 MPC 综述指出，近年的 AGV 研究已经把 MPC 从单纯路径跟踪扩展到运动规划、规划控制一体化、危险工况避障、动态交通参与者预测等任务。([link.springer.com](https://link.springer.com/article/10.1007/s43684-021-00005-z))

**四、MPC 的几种重要类型**

**1. 线性 MPC**

如果模型是线性的：

```text
x_{k+1} = A x_k + B u_k
```

目标函数是二次型，约束是线性的，那么 MPC 问题通常可以变成 QP，也就是二次规划。

优点是计算快、稳定性理论成熟、容易实时实现。

缺点是对强非线性系统不够准确，比如高速车辆侧偏、无人机大姿态运动、AUV 六自由度耦合运动。

**2. 非线性 MPC，NMPC**

如果模型是非线性的：

```text
x_{k+1} = f(x_k, u_k)
```

就是 NMPC。路径规划中只要加入圆形/椭圆障碍物约束、非线性车辆模型、AUV 水动力模型，就很容易变成 NMPC。

优点是模型真实、表达能力强。

缺点是求解慢，可能陷入局部最优，对初值和求解器依赖较强。

例如 Park 等 2009 年的自动车避障论文使用非线性 MPC 生成安全轨迹，再用单独控制器跟踪轨迹；论文明确提到使用简化车辆动力学在预测时域内预测状态，并把局部障碍物信息加入性能指标。([journals.sagepub.com](https://journals.sagepub.com/doi/10.1243/09544070JAUTO1149))

**3. 鲁棒 MPC / Tube-MPC**

现实系统模型不准，传感器有噪声，障碍物预测也有误差。鲁棒 MPC 会考虑不确定性，让轨迹不仅在理想情况下安全，而且在扰动范围内也安全。

Tube-MPC 的直觉是：不只规划一条线，而是规划一条“安全管道”。真实轨迹只要始终被包在管道内，就认为安全。

**4. 随机 MPC / Chance-Constrained MPC**

如果障碍物未来位置不是确定的，而是概率分布，就可以用机会约束：

```text
P(collision) ≤ ε
```

也就是允许极小概率的风险，但总体保持安全。这在自动驾驶、动态避障、多船避碰中很常见。

**5. 学习型 MPC**

学习型 MPC 会用历史经验改进模型、代价函数或终端集合。Rosolia 和 Borrelli 的 Learning MPC 论文提出从以往迭代轨迹中构造安全集和终端代价，从而保证递归可行性，并让性能随迭代不下降。([arxiv.org](https://arxiv.org/abs/1702.07064))

这类方法和强化学习、模仿学习、神经网络预测模型结合得越来越多，是近几年重要趋势。

**五、MPC 在路径规划中的关键理论问题**

**1. 可行性**

MPC 每一步都要解优化问题，但如果约束太严格，可能“无解”。例如机器人离障碍物太近，安全距离约束、速度约束、转向约束同时满足不了，就会不可行。

常见处理方式：

- 加松弛变量，把硬约束变成软约束。
- 设置终端集合，保证预测末端仍在可控安全区域。
- 使用安全走廊或可达集缩小搜索空间。
- 用上一步解作为 warm start，提高收敛稳定性。

**2. 递归可行性**

递归可行性指的是：如果当前时刻 MPC 有解，那么执行第一个控制后，下一时刻仍然应该有解。

这对路径规划很重要，因为不能只保证“现在看起来安全”，还要保证下一步不会把系统送进死胡同。Mayne 等人的 MPC 稳定性综述重点讨论的就是约束 MPC 中稳定性、最优性和递归可行性的理论基础。([colab.ws](https://colab.ws/articles/10.1016%2FS0005-1098%2899%2900214-9))

**3. 稳定性**

MPC 不是天然稳定的。为了保证稳定，常见设计包括：

- 终端代价 `V_f(x_N)`
- 终端约束 `x_N ∈ X_f`
- 局部稳定控制律
- Lyapunov 约束

2026 年一篇自动驾驶轨迹跟踪论文就把控制 Lyapunov 函数作为显式约束嵌入 MPC，用来同时保证轨迹跟踪、输入约束和闭环稳定性；该文还提到用稀疏 QP 和 warm-started OSQP 实现 50 ms 采样周期。([journals.sagepub.com](https://journals.sagepub.com/doi/10.1177/09544070261423602))

**4. 实时性**

路径规划的 MPC 必须实时求解。无人车可能 20 Hz 到 100 Hz 控制，无人机更快，AUV 虽然慢一些，但模型更复杂、环境不确定性更强。

实时性取决于：

- 预测时域 `N` 多长。
- 模型是线性还是非线性。
- 障碍物约束数量多少。
- 求解器是否高效。
- 是否使用 warm start。
- 是否把非凸问题转成凸近似。

2020 年 UAV 动态避障 NMPC 论文使用 PANOC/OpEn 求解器，采样时间为 50 ms，预测时域为 2 s，并将其定位为一种局部路径规划器。([arxiv.org](https://arxiv.org/abs/2008.00792))

**5. 安全约束建模**

路径规划中，障碍物约束通常写成：

```text
||p_robot - p_obstacle|| ≥ r_safe
```

但这个约束是非凸的，多个障碍物时更复杂。常见处理包括：

- 把障碍物距离放进代价函数，而不是硬约束。
- 对障碍物约束线性化。
- 构造安全走廊。
- 使用控制障碍函数 CBF。
- 使用椭圆约束、速度障碍 VO、碰撞锥等方法。
- 使用 Tube-MPC 或 chance constraint 处理不确定障碍物。

**六、按时间看 MPC 路径规划应用发展**

| 时间阶段 | 发展特点 | 代表论文/方向 |
|---|---|---|
| 2000 年左右 | MPC 理论成熟化，重点是稳定性、递归可行性、终端约束 | Mayne 等 2000 年《Constrained model predictive control: Stability and optimality》系统总结约束 MPC 理论。([colab.ws](https://colab.ws/articles/10.1016%2FS0005-1098%2899%2900214-9)) |
| 2005-2009 年 | MPC 开始进入自动驾驶转向控制、轨迹跟踪、避障 | Falcone 等 2007 年将 MPC 用于自动车主动前轮转向，在冰雪路面高速跟踪轨迹，并比较 NMPC 与在线线性化 MPC。([researchgate.net](https://www.researchgate.net/publication/3332878_Predictive_Active_Steering_Control_for_Autonomous_Vehicle_Systems)) Park 等 2009 年用 NMPC 生成避障轨迹。([journals.sagepub.com](https://journals.sagepub.com/doi/10.1243/09544070JAUTO1149)) |
| 2010-2016 年 | 从单车路径跟踪扩展到在线运动规划、复杂约束、移动机器人 | MPC 逐渐从“控制器”变成“局部规划器”，开始处理障碍物、安全区域、车辆动力学约束。 |
| 2017-2020 年 | 多智能体、无人机、鲁棒避障、学习 MPC 兴起 | Kamel 等 2017 年研究多架微型飞行器的去中心化 NMPC 避碰，考虑状态估计和其他智能体位置速度不确定性。([arxiv.org](https://arxiv.org/abs/1703.01164)) Rosolia 和 Borrelli 2017 年推动 Learning MPC。([arxiv.org](https://arxiv.org/abs/1702.07064)) 2020 年 UAV 动态障碍 NMPC 实现 50 ms 实时局部规划。([arxiv.org](https://arxiv.org/abs/2008.00792)) |
| 2021-2023 年 | 综述增多，MPC 成为自动驾驶、无人机、AUV/USV 的主流约束规划方法之一 | 自动地面车辆综述指出 MPC 已广泛用于 AGV，并扩展到规划控制一体化。([link.springer.com](https://link.springer.com/article/10.1007/s43684-021-00005-z)) MAV 综述总结了线性/非线性 MPC、预测时域调参、扰动观测、强化学习结合等趋势。([arxiv.org](https://arxiv.org/abs/2011.11104)) AUV 方向出现 IIFDS-NMPC 这类上层规划 + 下层 NMPC 跟踪结构，并在 BlueRov2 上做实物实验。([mdpi.com](https://www.mdpi.com/2077-1312/11/10/2014)) |
| 2024-2026 年 | 趋势转向安全证明、数据驱动、Koopman、CLF/CBF、实时优化 | 2025 年 Koopman MPC 尝试把非线性路径规划问题提升到 Koopman 空间中转成更快的 QP，并声称比原始 NMPC 快很多。([arxiv.org](https://arxiv.org/abs/2510.02584)) 2026 年 MPC-CLF 把 Lyapunov 稳定约束直接嵌入 MPC，提高稳定性可解释性。([journals.sagepub.com](https://journals.sagepub.com/doi/10.1177/09544070261423602)) |

**七、不同平台上的 MPC 应用特点**

**自动驾驶车辆**

自动驾驶是 MPC 路径规划最成熟的方向之一。原因是车辆有明确动力学约束：转角、转角速度、轮胎侧偏、道路边界、车道线、障碍车预测都可以自然写进 MPC。

自动驾驶中的 MPC 常用于：

- 路径跟踪
- 车道保持
- 换道
- 紧急避障
- 速度规划
- 轨迹规划与控制一体化

Falcone 2007 年论文是很经典的早期代表，它把 MPC 用于主动前轮转向，并在低附着冰面高速条件下做了仿真和实验。([researchgate.net](https://www.researchgate.net/publication/3332878_Predictive_Active_Steering_Control_for_Autonomous_Vehicle_Systems))

**移动机器人**

移动机器人中 MPC 主要解决两个问题：非完整约束和局部避障。

差速机器人、Ackermann 车、全向机器人都存在不同运动约束。MPC 可以把这些约束写入预测模型，比单纯人工势场更不容易产生不可执行动作。

但移动机器人环境障碍物多，MPC 容易遇到非凸约束和局部最优问题，所以常与 A*、RRT、人工势场、安全走廊结合。

**无人机 UAV / MAV**

无人机的难点是三维运动、姿态动力学复杂、控制频率高。MPC 的优势是能统一考虑动力学约束和避障约束，缺点是 NMPC 求解压力大。

Kamel 等 2017 年多 MAV 避碰论文很典型：它不要求事先给出无碰撞参考轨迹，而是用模型控制同时做轨迹跟踪和碰撞避免，并考虑其他飞行器位置速度不确定性。([arxiv.org](https://arxiv.org/abs/1703.01164))

**水面/水下机器人 USV/AUV**

USV/AUV 的 MPC 很适合你的研究方向。原因是水下/水面机器人具有明显约束：

- 惯性大，不能急停急转。
- 水动力模型非线性强。
- 洋流扰动明显。
- 传感器信息不完整。
- 安全距离和避障约束重要。
- AUV 能量有限，控制代价很关键。

2018 年 USV 论文把有限控制集 MPC 用于无人水面艇避碰，并强调它把路径规划和控制系统结合起来，输出更接近实际执行器的推进器转速和推进角，而不是抽象力/力矩。([mdpi.com](https://www.mdpi.com/2076-3417/8/6/926))

2023 年 AUV 论文采用上层 IIFDS 路径规划、下层 NMPC 轨迹跟踪的混合结构，并在 BlueRov2 平台上做了实物测试，说明 MPC 已经不只是仿真方法，而是在水下机器人实物验证中越来越常见。([mdpi.com](https://www.mdpi.com/2077-1312/11/10/2014))

**八、MPC 与深度强化学习/GNN 的关系**

如果你后续想把 MPC 和你前面提到的 AUV、GNN、MA-SAC 结合，可以这样理解：

MPC 的强项是：

- 安全约束明确
- 动力学可解释
- 控制输入可执行
- 适合做局部实时规划
- 能处理硬约束

强化学习的强项是：

- 可从交互中学习策略
- 适合复杂环境经验积累
- 可以处理难以手工建模的决策
- 推理时速度快

GNN 的强项是：

- 建模多智能体/障碍物之间的关系
- 适合可变数量邻居
- 适合 AUV 群体协同避障
- 能表达拓扑结构和局部交互

比较好的结合路线是：

```text
GNN：编码 AUV-障碍物-目标-邻居关系
        ↓
RL / MA-SAC：输出高层意图、参考速度、目标点或代价权重
        ↓
MPC / NMPC：在动力学和安全约束下生成可执行控制量
```

也就是说，**不要让强化学习直接控制推进器**，而是让它给 MPC 提供高层决策；MPC 负责最后一层安全可执行控制。这种结构在论文趋势上也很吻合：近年 MPC 正在向学习型、数据驱动、安全约束增强方向发展。

**九、推荐你优先读的论文清单**

1. Mayne, Rawlings, Rao, Scokaert, 2000  
   **Constrained Model Predictive Control: Stability and Optimality**  
   用途：理解 MPC 稳定性、递归可行性、终端约束的理论根基。  
   来源：[CoLab / DOI 信息](https://colab.ws/articles/10.1016%2FS0005-1098%2899%2900214-9) ([colab.ws](https://colab.ws/articles/10.1016%2FS0005-1098%2899%2900214-9))

2. Falcone et al., 2007  
   **Predictive Active Steering Control for Autonomous Vehicle Systems**  
   用途：自动驾驶中 MPC 路径跟踪/主动转向的经典早期应用。  
   来源：[ResearchGate 条目](https://www.researchgate.net/publication/3332878_Predictive_Active_Steering_Control_for_Autonomous_Vehicle_Systems) ([researchgate.net](https://www.researchgate.net/publication/3332878_Predictive_Active_Steering_Control_for_Autonomous_Vehicle_Systems))

3. Park et al., 2009  
   **Obstacle Avoidance of Autonomous Vehicles Based on Model Predictive Control**  
   用途：理解 NMPC 如何做在线避障轨迹生成。  
   来源：[SAGE Journals](https://journals.sagepub.com/doi/10.1243/09544070JAUTO1149) ([journals.sagepub.com](https://journals.sagepub.com/doi/10.1243/09544070JAUTO1149))

4. Rosolia & Borrelli, 2017  
   **Learning Model Predictive Control for Iterative Tasks**  
   用途：理解学习型 MPC，尤其是如何从历史轨迹构造安全集和终端代价。  
   来源：[arXiv:1702.07064](https://arxiv.org/abs/1702.07064) ([arxiv.org](https://arxiv.org/abs/1702.07064))

5. Kamel et al., 2017  
   **Nonlinear Model Predictive Control for Multi-Micro Aerial Vehicle Robust Collision Avoidance**  
   用途：理解多智能体/多无人机 NMPC 避碰。  
   来源：[arXiv:1703.01164](https://arxiv.org/abs/1703.01164) ([arxiv.org](https://arxiv.org/abs/1703.01164))

6. Sun et al., 2018  
   **Collision Avoidance Using Finite Control Set Model Predictive Control for Unmanned Surface Vehicle**  
   用途：理解 USV 中 MPC 如何直接输出实际执行器控制量。  
   来源：[Applied Sciences / MDPI](https://www.mdpi.com/2076-3417/8/6/926) ([mdpi.com](https://www.mdpi.com/2076-3417/8/6/926))

7. Nguyen et al., 2020  
   **Model Predictive Control for Micro Aerial Vehicles: A Survey**  
   用途：系统了解无人机 MPC 的线性/非线性、鲁棒性、RL 结合趋势。  
   来源：[arXiv:2011.11104](https://arxiv.org/abs/2011.11104) ([arxiv.org](https://arxiv.org/abs/2011.11104))

8. Du, Zhou, Arai, 2023  
   **Hybrid Layer of IIFDS and NMPC for Navigation and Control of AUVs**  
   用途：与你的 AUV 动态避障方向最相关，适合理解“上层路径规划 + 下层 NMPC 跟踪”。  
   来源：[JMSE / MDPI](https://www.mdpi.com/2077-1312/11/10/2014) ([mdpi.com](https://www.mdpi.com/2077-1312/11/10/2014))

9. Gao et al., 2026  
   **A Lyapunov-Constrained MPC Framework for Stable Trajectory Tracking of Autonomous Vehicles**  
   用途：理解最新趋势中如何把稳定性证明显式放进 MPC 优化约束。  
   来源：[SAGE Journals](https://journals.sagepub.com/doi/10.1177/09544070261423602) ([journals.sagepub.com](https://journals.sagepub.com/doi/10.1177/09544070261423602))

总结一句：**MPC 路径规划的发展主线，是从“约束控制理论”走向“动态环境中的实时安全轨迹优化”，再走向“与学习、概率预测、图结构、多智能体协同相结合”。** 对 AUV 动态避障来说，MPC 非常适合作为最后一层安全控制器，而 GNN/强化学习更适合作为环境关系建模和高层决策模块。




















The manuscript can be visited at [arXiv](https://arxiv.org/abs/2412.02942) or [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253525007171)

[The code](https://github.com/GeoX-Lab/STDCformer).
