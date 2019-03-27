---
title: 关于transformer结构的一些小Tips
layout: post
categories: NLP
tags: NLP 算法
excerpt: 关于transformer的详细图文详解教程
---
#### Transformer来源
Transformer是谷歌在2017年做机器翻译任务的**“Attention is all you need”**的论文中提出的，Transformer的结构包含两部分：Encoder和Decoder。Encoder是六层编码器首位相连堆砌而成，Decoder也是六层解码器堆成的。其结构如下：
![Image Title](https://i.loli.net/2019/03/27/5c9b3779d79b9.png)

详细的结构图如：
![Image Title](https://i.loli.net/2019/03/27/5c9b370ec551f.jpg)

论文里的原图如下：
![Image Title](https://i.loli.net/2019/03/27/5c9b4093ae2c0.jpg)

#### Transformer结构
对于Transformer结构来说，主要就是Encoder和Dcoder结构组成，每一个Encoder是由self-attention+Feed Forward NN构成，如下图所示，所以我们首先要理解self-attention。
![Image Title](https://i.loli.net/2019/03/27/5c9b3e1be49ac.png)
![Image Title](https://i.loli.net/2019/03/27/5c9b3e1be641e.jpg)

每一个Decoder是由Self-Attention+Encoder-Decoder Attention+Feed Forward NN构成，结构如下图所示：
![Image Title](https://i.loli.net/2019/03/27/5c9b3ee32fe81.png)

#### Encoder结构详解
一般的博客都是先将重要的Self-Attention讲起，但是我觉得还是应该从输入讲起，一步一步循序渐进就好，首先讲一下Transformer的输入，其输入是词向量的嵌入和位置编码向量的相加，这样就可以把位置信息考虑进去，论文中的位置嵌入公式是：
![Image Title](https://i.loli.net/2019/03/27/5c9b426ec0116.jpg)

pos代表的是位置，i代表的是维度，偶数位置的文字会透过sin函数进行转换，奇数位置的文字则透过cos函数进行转换，藉由三角函数，可以发现positional encoding 是个有週期性的波长；举例来说，[pos+k]可以写成PE[pos]的线性转换，使得模型可以学到不同位置文字间的相对位置。
![Image Title](https://i.loli.net/2019/03/27/5c9b4314074d9.jpg)

代码为：
```
def GetPosEncodingMatrix(max_len, d_emb):
    '''
    PE(pos, 2i) = sin( pos /(1000_(2i /d_model) )
    PE(pos, 2i+1) = cos( pos /(1000_(2i /d_model))
    取整除 - 返回商的整数部分（向下取整） (j // 2)
    通常位置编码是一个长度为 d_{model} 的特征向量，这样便于和词向量进行单位加的操作
    pos_enc[1:, 0::2] 偶数# 按步长为2取第二维的索引0到末尾之间的元素，也就是第一列和第三列
    1::2  奇数
    :param max_len:句子长度
    :param d_emb:词向量的维度
    :return:
    '''
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc

```



#### 参考

[放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)

[从Seq2seq到Attention模型到Self Attention（二）](https://wallstreetcn.com/articles/3417279)

[NLP学习笔记（一）：图解Transformer+实战](https://blog.csdn.net/linxid/article/details/84321617)
