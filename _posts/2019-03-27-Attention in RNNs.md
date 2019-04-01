---
title: Attention in RNNs 机制图解
layout: post
categories: 算法
tags: 深度学习
excerpt: 关于 Attention in RNNs 机制图解
---
#### 文章来源

由于文章的链接打不开，就在爱可可老师的微博下载了pdf版本，但出于对作者的敬意还是贴出了网址https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05

![Image Title](https://i.loli.net/2019/03/29/5c9d7712b86f5.png)

####RNN
RNN作为处理序列数据的方法，已经在很多方面取得了成功，如：机器翻译，情感分析，图像描述....。为了解决RNN出现的一些问题，如vanishing gradients，出现了一些变体LSTM，GRU等。然而，即使是更高级的模型也有其局限性，研究人员在处理长数据序列时也很难开发出高质量的模型。例如，在机器翻译中，RNN必须找到由几十个单词组成的长输入和输出句子之间的联系。现有的RNN体系结构似乎需要改变和适应，以便更好地处理这些任务。
Attention机制如下：

![Image Title](https://i.loli.net/2019/03/29/5c9d7afce7e03.png)

从上图可以看到s_i的输入是拼接向量[s_{i-1},h_i]，通过softmax得到[0,1]之间的权重，然后乘以h_i.整个网络如下：

![Image Title](https://i.loli.net/2019/03/29/5c9d7afce9fbc.png)

RNN的Attention如下：

![Image Title](https://i.loli.net/2019/03/29/5c9d7afd03e84.png)

下面是分解讲解the attention weights and context vectors：
![Image Title](https://i.loli.net/2019/03/29/5c9d7afd0d794.png)

第一步是由编码器计算向量h1、h2、h3、h4。这些向量都作为attention机制的输入向量。在这里，解码器首先通过输入其初始状态向量s0进行处理，我们得到了第一个注意力输入序列[s0, h1]， [s0, h2]， [s0, h3]， [s0, h4]。注意机制计算权重α11第一组的权重值,α12,α13,α14启用第一个上下文向量的计算c1。解码器现在使用[s0,c1]并计算第一个RNN输出y1。

![Image Title](https://i.loli.net/2019/03/29/5c9d7afd0cd2d.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b708aa.png)

在接下来的时间步长中，注意机制输入序列[s1, h1]， [s1, h2]， [s1, h3]， [s1, h4]。

![Image Title](https://i.loli.net/2019/03/29/5c9d7afd0e0ae.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b7235e.png)
剩下的与前面一样：

![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b70034.png)
在下一步，注意机制有一个输入序列[s2, h1]， [s2, h2]， [s2, h3]， [s2, h4]。

![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b71126.png)


![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b717d2.png)



![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b72a8f.png)
