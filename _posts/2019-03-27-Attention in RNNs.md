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
RNN作为处理序列数据的方法，已经在很多方面取得了成功，如：机器翻译，情感分析，图像描述....。为了解决RNN出现的一些问题，如vanishing gradients，出现了一些变体LSTM，
GRU等。然而，即使是更高级的模型也有其局限性，研究人员在处理长数据序列时也很难开发出高质量的模型。例如，在机器翻译中，RNN必须找到由几十个单词组成的长输入和输出句子之间的联系。现有的RNN体系结构似乎需要改变和适应，以便更好地处理这些任务。

![Image Title](https://i.loli.net/2019/03/29/5c9d7afce7e03.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7afce9fbc.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7afd03e84.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7afd0cd2d.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7afd0d794.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7afd0e0ae.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b70034.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b708aa.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b71126.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b717d2.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b7235e.png)
![Image Title](https://i.loli.net/2019/03/29/5c9d7b0b72a8f.png)
