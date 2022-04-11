**华东师范大学数据科学与工程学院实验报告** 

| **课程名称**：统计学习与机器方法      | **年级**：2019     | **实践成绩**：        |
| ------------------------------------- | ------------------ | --------------------- |
| **指导教师**：董启文                  | **姓名**：周辛娜   | **学号**：10195501442 |
| **上机实践**： finalproject：人脸识别 | **上机实践时间**： | 2021.12.25            |



## 任务：

任务1：使用机器学习进行人类分类识别，给出识别准确率。

任务2：使用聚类或分类算法发现表情相似的脸图。



## 数据集：

来源于CMU Machine Learning Faces，[Neural Networks for Face Recognition (cmu.edu)](http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/faces.html)，里面包含20个人，每人32张脸图(含表情)。



## 一、探索数据集

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211222231046833.png" alt="image-20211222231046833" style="zoom:67%;" />

### 1.得到name，direction，expression，wear_sunglasses

下载的数据集即faces_4如上图所示，其中包含了20个人的不同图片，进入第一个文件夹an2i，里面共有32张图片，图片的文件名表名了这张图片记录的是这个人的名字、方向direction、表情、有没有戴sunglasses，那么可以根据这个进行数据整理。如an2i_straight_neutral_open_4.pgm就可以被拆成an2i，straight，neutral，open，4.pgm，进而遍历faces_4中的每个人的32个文件名，得到4个分类标签，分别是name，direction，expression，wear_sunglasses，其中name，代表的人名，共有20类，direction代表脸的朝向，共有4类，expression代表表情，共有4类，wear_sunglasses代表有没有戴sunglasses，有2类。

最后得到的name，direction，expression，wear_sunglasses如下图所示：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211229231233718.png" alt="image-20211229231233718" style="zoom:150%;" />



### 2. 构造可以被用来训练的数据

这里借鉴了fetch_lfw_people这一个sklearn中自带的人脸数据集，它的返回对象sklearn.utils.Bunch，该对象类似于字典，分为5个属性：

'data'，'images'，'target'，'target_names'，'DESCR'。

这里我构造的数据分为5个属性：

data 图像特征、name、direction、expression、wear_sunglasses。

采用字典的方式。

这里一个重要的函数就是construct_data()

![image-20211230083854392](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211230083854392.png)

而且还要将对应的数字对应到对应的类别，而不应该是字符串。



最后得到的数据如下图所示：









<img src="C:\Users\asus\Desktop\1.png" style="zoom: 80%;" />



可以看到，name有20个类别，所以取值范围是0-19，direction有4个类别，取值范围是0-3，expression有4个类别，取值范围0-3，wear_sunglasses有2个类别，取值分别是0,1。







## 二、数据预处理及训练集测试集划分

由于是顺序读取，故需要random.shuffle打乱数据，然后通过定义的construct_data函数分别得到训练集和测试集，这里选择前305个为训练集，剩下的图片为测试集。



在一开始进行训练的时候，并没有意识到需要用random.shuffle打乱数据，造成训练集的种类全集不到20类，而测试集的种类全集不到20类，而且之间的重合项很少，所以在最后模型的实验结果非常差，最后才发现是在划分训练集合测试集的时候没有考虑的样本均衡的问题，经过random.shuffle打乱后的数据集经过划分后，训练集合测试集的种类全集都有20类了，而且分布相对比较均匀了。



## 三、任务一，人脸分类识别

第一个任务：使用机器学习进行人脸分类识别，给出识别准确率，其实name就是对应的标签



### 使用两种机器学习方法完成

分别是SVM和神经网络，SVM采用的kernel为rbf，根据网格搜索选取最优超参数，神经网络借助tensorflow搭建模型。

支持向量机(SVM) 在有限的数据量下性能非常好。给定训练数据，SVM将得到一个最优超平面，从而对训练数据进行分类。本实验中了，是调用了sklearn中的库，即from sklearn.svm import SVC



### 1）SVM模型

#### 通过网格搜索选取最佳超参数：

![image-20211230084609114](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211230084609114.png)

得到C=1



#### 代入模型，进行训练

仅仅使用SVM效果就很好了，如果再加上PCA降维准确率反而降低了，最后就直接使用SVM，至于为什么会降低，会在解释中阐述。



#### 实验结果：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211229232335790.png" alt="image-20211229232335790" style="zoom:80%;" />

最终准确率达到99%，效果不错



### 2) 神经网络

这里使用models.Sequential()来搭建神经网络，使用model.compile()方法来配置训练方法，使用model.fit()方法来执行训练过程。

采用一层全连接层，20个神经元，softmax激活函数，L2正则化，使用随机梯度下降SGD，学习率为0.009，使用交叉熵损失函数。



#### 训练过程:

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211229232822264.png" alt="image-20211229232822264" style="zoom:80%;" />







训练好了以后，进行预测，由于人名有20类，最后网络输出的是独热向量编码，需要将它转变成对应的第几类，



#### 最终得到的实验结果：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211229232743304.png" alt="image-20211229232743304" style="zoom:80%;" />

最后达到97%的准确率，效果不错



## 四、任务二，发现表情相似的图

expression就是对应的标签



### 使用三种方法，分别是SVM，神经网络和kmeans



### 1）SVM

#### 通过网格搜索选取最佳超参数：

![image-20211230084821140](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211230084821140.png)

得到C=0.1







#### 代入模型，进行训练



#### 实验结果：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211229233013045.png" alt="image-20211229233013045" style="zoom:80%;" />

最后准确率才23%，效果不佳，基本上就等于盲猜了。



### 2）神经网络

这里使用models.Sequential()来搭建神经网络，使用model.compile()方法来配置训练方法，使用model.fit()方法来执行训练过程，采用一层全连接层，4个神经元，softmax激活函数，l2正则化，使用随机梯度下降SGD，学习率为0.01，使用交叉熵损失函数。



#### 训练过程:

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211229233303668.png" alt="image-20211229233303668" style="zoom:80%;" />

可以看到loss history波动非常大，而且下降幅度也并不大，classification accuracy这一块，不管是在测试集上还是验证集上，准确率都是波动非常大，而且基本上就是在盲猜。



训练好了以后，进行预测，由于表情有4类，最后网络输出的是独热向量编码，需要将它转变成对应的第几类。



#### 最终得到的实验结果：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211229233400565.png" alt="image-20211229233400565" style="zoom:80%;" />

最后效果很差，准确率才18%。



### 3）kmeans 

聚为4类，表情有4类。



#### 实验结果：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211229233442568.png" alt="image-20211229233442568" style="zoom:80%;" />

最后准确率才24%，效果不佳，基本上等于盲猜。



### 解释：

这三种方法，SVM，神经网络，kmeans聚类算法，不管是分类方法还是聚类方法效果都很差，和盲猜差不多，盲猜的概率是0.25，而有的甚至比盲猜表现还差。

猜测主要原因是在于数据集本身质量不是很好，该数据集年代较为久远，它不仅包含人脸，还包含了背景，会有很大的噪声。该数据集规模小，损失了很多人脸的特征，并且是两通道的灰度图。

这也解释了为什么经过PCA降维后再进行SVM，分类的准确率反而降低了，因为本身数据就没有包含很多特征，降维使得特征更少了。









## 五、进行朝向和有没有戴墨镜的区别

### 1）区别人脸朝向：

#### 通过网格搜索选取最佳超参数：

![image-20211230084856426](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211230084856426.png)

得到C=10



#### 代入模型，进行训练



#### 最后得到的实验结果：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211229234152788.png" alt="image-20211229234152788" style="zoom:80%;" />

最后准确率达到97%，效果不错

#### 可视化查看朝left的前5个样本：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211230091649237.png" alt="image-20211230091649237" style="zoom:150%;" />

#### 可视化查看朝right的前5个样本：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211230091804645.png" alt="image-20211230091804645" style="zoom:150%;" />

#### 可视化查看朝straight的前5个样本：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211230091842390.png" alt="image-20211230091842390" style="zoom:150%;" />

#### 可视化查看朝up的前5个样本：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211230091917373.png" alt="image-20211230091917373" style="zoom:150%;" />

可以看到，分类效果不错。



### 2）区分人脸是否戴sunglasses

使用SVM



#### 最后得到的实验结果：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211229234241261.png" alt="image-20211229234241261" style="zoom:80%;" />

准确率达到91%，效果不错



#### 可视化查看被分类到戴sunglasses的前5个样本：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211230091302228.png" alt="image-20211230091302228" style="zoom:150%;" />

#### 可视化查看被分类到没戴sunglasses的前5个样本：

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20211230091211260.png" alt="image-20211230091211260" style="zoom:150%;" />

可以看到，分类效果不错。







## 六、总结

​	本次实验针对CMU Machine Learning Faces数据集，使用了两种机器学习方法完成任务一（人脸分类识别），准确率再97%以上，效果很不错，使用三种方法完成任务二（发现表情相似的图），但是最后的效果不太好，基本等同于盲猜，猜测这主要是同数据集本身的质量不高，特征少，噪声大，样本数量少有关。此外，还使用SVM区别了人脸朝向和人脸是否戴sunglasses，最后的分类准确率都在90%以上，效果不错。