# 传统指纹识别方法

## 一、介绍

传统指纹识别算法步骤为：

* 指纹图像增强 

* 二值化
* 细化 
* 提取特征点，计算特征点描述子 
*  匹配

这五步符合传统模式识别的标准流程——数据预处理、提取特征、分类。



## 二、具体过程

首先对指纹图像进行归一化，然后进行脊线方向场计算，进行Gabor滤波增强，二值化得到二值图像，最终效果如下图所示：

<p align="center"><img src="./resources/bw.png" width=30%></p>

<h6 align='center'>原图与二值化图像对比</h6>



接着从二值化图像中提取Harris角点，并计算Harris角点的ORB特征描述子，并进行特征匹配。最后使用距离判别条件——若匹配特征描述子之间的平均距离小于一个阈值则认为匹配成功，否则则认为失败。

<p align="center"><img src="./resources/match.png" width=30%></p>

<h6 align='center'>特征点匹配</h6>

## 三、结果

用测试数据(9982正样本对，15964个负样本对)对模型进行评估，得到PR曲线如下所示。可以看到随着查全率(recall)的增长，查准率(precision)维持在1附近，当查全率接近1附近时查准率降到75%。说明用Harris角点的ORB特征描述子可以较好的区别正负样本，正负样本之间只有少量的交叠。

<p align="center"><img src="./result/pr-curve_tradition.png" width="50%"></p>

取匹配点描述子平均Hamming距离为2.78作为正样本和负样本之间的判别边界(平均距离小于2.78为正样本，否则为负样本)，得到混淆矩阵如下所示

<p align="center"><img src="./result/confusion_matrix_tradition.png" width="50%"></p>

其中Accuracy = 79.04%, Precision = 99.98%, Recall = 45.53%,  F1=62.57%。
