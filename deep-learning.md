# 指纹识别深度学习方法
## 一、模型
本次实验使用Siamese网络架构，构造正负样本对进行训练，将样本嵌入(embedding)到一个度量空间，使得相同语义(同一人的同一手指指纹)的样本靠近，不同语义的样本远离。

<p align="center"><img  src="./resources/siamese_network.jpg" width="35%" height="35%"></p>



网络具体架构如下图所示，由4层卷积池化层加4层全连接层构成。其中每次卷积都采用3 x 3卷积核，s = 1, p = 2。卷积层都采用CBR结构(Conv + BN + ReLU)，
池化操作为平均池化(k = 2, s = 2, p = 0)，最终通过全连接层将数据嵌入到一个30维的向量空间中。



<p align="center"><img  src="./resources/network_structure.png" width="75%" height="75%"></p>

## 二、损失函数
损失函数采取了LeCun于2006年提出的对比损失函数(Contrastive Loss Function)<sup>[1]</sup>，该损失函数的设计思想是：

* 近似样本之间的距离越小越好。
* 不相似样本之间的距离如果小于m，则相互排斥使其距离接近m。

可以用下图形象地表示该损失函数的作用效果。



<p align="center"><img  src="resources/contrastive_loss_function.png" width="25%" height="25%"></p>



## 三、结果

用测试数据(9982正样本对，15964个负样本对)对模型进行评估，得到PR曲线如下所示。可以看到查准率(Precision)在不同查全率(Recall)上都是100%。说明正样本对和负样本对距离之间存在一个界(margin)，正负样本之间不存在交叠的情况。当取距离大于1.148为负样本、距离小于1.148为负样本时可以完美地分离测试集正负样本(precision=100%, recall=100%)。

<p align="center"><img src="./result/pr-curve.png" width=45%></p>

取阈值为1.99模型对应混淆矩阵如下所示：

<p align="center"><img  src="result/confusion_mat.png" width="45%" height="45%"></p>



其中Precision = 99.87%， Recall=100%, Accuracy = 99.87%。以下是将负样本对错分成正样本对的32个样本对：

<p align="center"><img src='result/0.jpg' width="50%"></p>

<p align="center"><img src='result/1.jpg' width=50%></p>

<p align="center"><img src='result/2.jpg' width=50%></p>

<p align="center"><img src='result/3.jpg' width=50%></p>

<h6 align="center">FP样本对</h6>

可以看到这些错分为正样本的指纹对在空间分布上具有较高的一致性，说明Siamese网络提取到了指纹空间分布的信息，此外可以发现这些错分的指纹对中至少有一方是包含噪声的，说明模型可能被噪声干扰，稳定性可以通过数据增强技术进一步提高。

**补充**：本次实验还使用了基于传统模式识别的指纹识别算法，可以在[traditonal-method.md](./traditional-method.md)中查看

## 四、参考资料

[1] [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf), 2006, Raia Hadsell, Sumit Chopra, Yann LeCun

