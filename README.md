# 指纹验证(fingerprint verifcation)
## 一、介绍
指纹识别即对人体指纹进行识别，该任务有两种理解方式：1. 由于标签离散，将指纹识别理解为分类任务(classification)，采用one-hot编码和交叉熵损失函数测度，这种方式一般针对封闭的指纹分类系统，
模型一旦训练好不能增加人数。 2. 将指纹识别理解成图像检索任务(image retrieval)，即给定两张指纹图片(其中一张是咨询(query)图片)，输出它们的相似性测度。这是指纹识别常用的方式，可以随时增减人，
常用于指纹验证(fingerprint verification)系统和指纹检索系统。本仓库实现第二种方式。

## 二、数据集
本次实验采用2018年发布的Sokoto Coventry指纹识别数据集(SOCO-Fing)<sup>[1]</sup>。SOCO-Fing的原始图片600名非洲人的6000张指纹图片组成，每人十个手指对应十张图片，
图片是96 x 103像素的灰度图像。除原始图片外，该数据集还对图片进行了一定的变换，包括Z字切割、湮没和中心旋转(如下图所示)。这些变换将任务分成了简单、中等、难三个层级，
数据集总共有55273张指纹图片。

![dataset](./resources/dataset.png)

SOCO-Fing数据集 [https://www.kaggle.com/ruizgara/socofing/home](https://www.kaggle.com/ruizgara/socofing/home) 

## 三、模型
本次实验使用Siamese网络架构，构造正负样本对进行训练，将样本嵌入(embedding)到一个度量空间，使得相同语义(同一人的同一手指)的样本靠近，不同语义的样本远离，相似性测度采用余弦距离。

## 四、损失函数
损失函数采取了LeCun于2006年提出的对比损失函数(Contrastive Loss Function)<sup>[2]</sup>，该损失函数的基本原则是：1. 近似样本之间的距离越小越好。2. 不相似样本之间的距离如果小于m，
则相互排斥使其距离接近m。可以形象地用下图表示：

![contrastive-loss-function](resources/contrastive_loss_function.png)
 

## 五、结果


## 六、参考资料
[1] Shehu Y I , Ruiz-Garcia A , Palade V , et al. Sokoto Coventry Fingerprint Dataset[J]. 2018.

[2] [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf), 2006, Raia Hadsell, Sumit Chopra, Yann LeCun