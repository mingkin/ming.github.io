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
然后就是万众瞩目的Multi-Head Attention结构，所谓多头attention就是多做几次attention，我们先讲attention，然后讲Mutil-Head，
Self-Attention步骤：

*1 将输入词转变成词向量，即得到Embedding层；
*2 每个词向量得到一个Query向量, Key 向量和 Value 向量（都一样）；

![Image Title](https://i.loli.net/2019/03/28/5c9cc1fb5bf46.jpg）

*3 为每一个词向量计算一个 score：query.dot(k) ；我们需要计算句子中的每一个词对当前词的score。这个score决定对句子的其他部分注意力是多少，也就是如何用句子的其他部分类表征当前词。

![Image Title](https://i.loli.net/2019/03/28/5c9cc254f2406.jpg）

在NLP的领域中，Key, Value通常就是指向同一个文字隐向量(word embedding vector)，可以参考key、value的起源论文 Key-Value Memory Networks for Directly Reading Documents。

![Image Title](https://i.loli.net/2019/03/28/5c9cbd56f422a.jpg)
![Image Title](https://i.loli.net/2019/03/28/5c9cbf05e5d9f.jpg)

将我们的词向量矩阵X 和权重矩阵W_Q,W_K,W_VW 相乘，即可得到Query 、Key 、Value 向量。
![Image Title](https://i.loli.net/2019/03/28/5c9cbf4ee8418.jpg)

接下来这张图可以清晰的说明白Query 、Key 、Value Value三个向量的关系。”The transformer”除以\sqrt{d_k}，目的是避免内积过大时，softmax得出的结果非0即1。

![Image Title](https://i.loli.net/2019/03/28/5c9cbfef329c0.jpg)

下面讲一下Mutil-Head，如果我们只计算一个attention，很难捕捉输入句中所有空间的讯息，为了优化模型，论文当中提出了一个新颖的做法：Multi-head attention，概念是不要只用d_{model}维度的key, value, query们做单一个attention，而是把key, value, query们线性投射到不同空间h次，分别变成维度d_{q}, d_{k} 和 d_{v}，再各自做attention，其中，d_{k}=d_{v}=d_{model}/h=64，概念就是投射到h个head上。

![Image Title](https://i.loli.net/2019/03/28/5c9cc3d17f03c.jpg）

”Transformer”用了8个attention head，所以我们会产生8组encoder/decoder，每一组都代表将输入文字的隐向量投射到不同空间，如果我们重复计算刚刚所讲的self-attention，我们就会得到8个不同的矩阵Z，可是呢，feed-forward layer期望的是一个矩阵而非8个，所以我们要把这8个矩阵拼接在一起，然后乘上一个权重矩阵，还原成一个矩阵Z。
![Image Title](https://i.loli.net/2019/03/28/5c9cc41425a25.jpg）

代码为：
```
class MultiHeadAttention():
	# mode 0 - big martixes, faster; mode 1 - more clear implementation
	def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
		self.mode = mode
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		self.dropout = dropout
		if mode == 0:
			self.qs_layer = Dense(n_head*d_k, use_bias=False)
			self.ks_layer = Dense(n_head*d_k, use_bias=False)
			self.vs_layer = Dense(n_head*d_v, use_bias=False)
		elif mode == 1:
			self.qs_layers = []
			self.ks_layers = []
			self.vs_layers = []
			for _ in range(n_head):
				self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
		self.attention = ScaledDotProductAttention(d_model)
		self.layer_norm = LayerNormalization() if use_norm else None
		self.w_o = TimeDistributed(Dense(d_model))

	def __call__(self, q, k, v, mask=None):
		d_k, d_v = self.d_k, self.d_v
		n_head = self.n_head

		if self.mode == 0:
			qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
			ks = self.ks_layer(k)
			vs = self.vs_layer(v)

			def reshape1(x):
				s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
				x = tf.reshape(x, [s[0], s[1], n_head, s[2]//n_head])
				x = tf.transpose(x, [2, 0, 1, 3])  
				x = tf.reshape(x, [-1, s[1], s[2]//n_head])  # [n_head * batch_size, len_q, d_k]
				return x
			qs = Lambda(reshape1)(qs)
			ks = Lambda(reshape1)(ks)
			vs = Lambda(reshape1)(vs)

			if mask is not None:
				mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
			head, attn = self.attention(qs, ks, vs, mask=mask)  
				
			def reshape2(x):
				s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
				x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
				x = tf.transpose(x, [1, 2, 0, 3])
				x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
				return x
			head = Lambda(reshape2)(head)
		elif self.mode == 1:
			heads = []; attns = []
			for i in range(n_head):
				qs = self.qs_layers[i](q)   
				ks = self.ks_layers[i](k) 
				vs = self.vs_layers[i](v) 
				head, attn = self.attention(qs, ks, vs, mask)
				heads.append(head); attns.append(attn)
			head = Concatenate()(heads) if n_head > 1 else heads[0]
			attn = Concatenate()(attns) if n_head > 1 else attns[0]

		outputs = self.w_o(head)
		outputs = Dropout(self.dropout)(outputs)
		if not self.layer_norm: return outputs, attn
		outputs = Add()([outputs, q])
		return self.layer_norm(outputs), attn

class ScaledDotProductAttention():
	def __init__(self, d_model, attn_dropout=0.1):
		self.temper = np.sqrt(d_model)
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k, v, mask):
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
		if mask is not None:
			mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
			attn = Add()([attn, mmask])
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output, attn
        
 ```
 
Multi-Head Attention的优点：

扩展模型能力可以注意到不同位置，一个注意力模型的关注点也许是错的，通过多个注意力模型可以提高这种泛化能力；
使得注意力层具有多个表示子空间，比如说上文的8个注意力模型，经过训练后，我们就可以将输入的词嵌入映射到8个不同的表示子空间；

构成Transformer的Encoder除了上述部分还有残差网络和一层归一化，如下图：

![Image Title](https://i.loli.net/2019/03/28/5c9cc5b98589f.jpg）

Residual connection 就是构建一种新的残差结构，将输出改写成和输入的残差，使得模型在训练时，微小的变化可以被注意到，这种架构很常用在电脑视觉(computer vision)，有兴趣可以参考神人Kaiming He的Deep Residual Learning for Image Recognition。

Layer normalization则是在深度学习领域中，其中一种正规化方法，最常和batch normalization进行比较，layer normalization的优点在於它是独立计算的，也就是针对单一样本进行正规化，batch normalization则是针对各维度，因此和batch size有所关联，可以参考layer normalization。

![Image Title](https://i.loli.net/2019/03/28/5c9cc68d351d8.jpg）

其代码为：
```
class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
									 initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
									initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape

```
Encoder/Decoder中的attention sublayers都会接到一层feed-forward networks(FFN)：两层线性转换和一个RELU，论文中是根据各个位置(输入句中的每个文字)分别做FFN，举例来说，如果输入文字是<x1,x2…xm>，代表文字共有m个。其中，每个位置进行相同的线性转换，这边使用的是convolution1D，也就是kernel size=1，原因是convolution1D才能保持位置的完整性，可参考CNN，模型的输入/输出维度d_{model}=512，但中间层的维度是2048，目的是为了减少计算量，这部分一样参考神人Kaiming He的Deep Residual Learning for Image Recognition。

![Image Title](https://i.loli.net/2019/03/28/5c9cc7874bf74.jpg)


```
class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = LayerNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)
```
Encoder 代码为：
```
class EncoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
	def __call__(self, enc_input, mask=None):
		output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
		output = self.pos_ffn_layer(output)
		return output, slf_attn

```

Encoder讲完后，接下来讲一下Decoder的结构。
![Image Title](https://i.loli.net/2019/03/28/5c9cc89f33bf5.jpg）

Decoder的运作模式和Encoder大同小异，也都是经过residual connections再到layer normalization。Encoder中的self attention在计算时，key, value, query都是来自encoder前一层的输出，Decoder亦然。不一样的是，为了避免在解码的时候，还在翻译前半段时，就突然翻译到后半段的句子，会在计算self-attention的softmax之前先mask掉未来的位置(设定成-∞)。这个步骤确保在预测位置i的时候只能根据i之前位置的输出，其实这个是对Encoder-Decoder attention 的特性而做的配套措施，因为Encoder-Decoder attention可以看到encoder的整个句子，“Encoder-Decoder Attention”和Encoder/Decoder self attention不一样，它的Query来自于decoder self-attention，而Key、Value则是encoder的output。

![Image Title](https://i.loli.net/2019/03/28/5c9cc9ff268b1.jpg）

其代码为：
```
class DecoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.enc_att_layer  = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
	def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None):
		output, slf_attn = self.self_att_layer(dec_input, dec_input, dec_input, mask=self_mask)
		output, enc_attn = self.enc_att_layer(output, enc_output, enc_output, mask=enc_mask)
		output = self.pos_ffn_layer(output)
		return output, slf_attn, enc_attn
        
 ```

其整体的运行方式为：从输入文字的序列给Encoder开始，Encoder的output变成attention vectors的Key、Value，然后传送至encoder-decoder attention layer，让Decoder将注意力放在输入文字序列的某个位置进行解码。
最后的 Linear and Softmax Layer，Decoder最后会产生一个向量，传到最后一层linear layer后做softmax。Linear layer只是单纯的全连接层网络，并产生每个文字对应的分数，softmax layer会将分数转成机率值，最高机率的值就是在这个时间顺序时所要產生的文字。

![Image Title](https://i.loli.net/2019/03/28/5c9ccbfa34b15.jpg）

其代码为：

```
class Transformer:
	def __init__(self, i_tokens, o_tokens, len_limit, d_model=256, 
			  d_inner_hid=512, n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1, 
			  share_word_emb=False):
		self.i_tokens = i_tokens
		self.o_tokens = o_tokens
		self.len_limit = len_limit
		self.src_loc_info = True
		self.d_model = d_model
		self.decode_model = None
		d_emb = d_model

		pos_emb = Embedding(len_limit, d_emb, trainable=False, 
						   weights=[GetPosEncodingMatrix(len_limit, d_emb)])

		i_word_emb = Embedding(i_tokens.num(), d_emb)
		if share_word_emb: 
			assert i_tokens.num() == o_tokens.num()
			o_word_emb = i_word_emb
		else: o_word_emb = Embedding(o_tokens.num(), d_emb)

		self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, 
                            word_emb=i_word_emb, pos_emb=pos_emb)
		self.decoder = Decoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, 
							word_emb=o_word_emb, pos_emb=pos_emb)
		self.target_layer = TimeDistributed(Dense(o_tokens.num(), use_bias=False))

	def get_pos_seq(self, x):
		mask = K.cast(K.not_equal(x, 0), 'int32')
		pos = K.cumsum(K.ones_like(x, 'int32'), 1)
		return pos * mask

	def compile(self, optimizer='adam', active_layers=999):
		src_seq_input = Input(shape=(None,), dtype='int32')
		tgt_seq_input = Input(shape=(None,), dtype='int32')

		src_seq = src_seq_input
		tgt_seq  = Lambda(lambda x:x[:,:-1])(tgt_seq_input)
		tgt_true = Lambda(lambda x:x[:,1:])(tgt_seq_input)

		src_pos = Lambda(self.get_pos_seq)(src_seq)
		tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
		if not self.src_loc_info: src_pos = None

		enc_output = self.encoder(src_seq, src_pos, active_layers=active_layers)
		dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, active_layers=active_layers)	
		final_output = self.target_layer(dec_output)

		def get_loss(args):
			y_pred, y_true = args
			y_true = tf.cast(y_true, 'int32')
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
			mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
			loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
			loss = K.mean(loss)
			return loss

		def get_accu(args):
			y_pred, y_true = args
			mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
			corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
			corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
			return K.mean(corr)
				
		loss = Lambda(get_loss)([final_output, tgt_true])
		self.ppl = Lambda(K.exp)(loss)
		self.accu = Lambda(get_accu)([final_output, tgt_true])

		self.model = Model([src_seq_input, tgt_seq_input], loss)
		self.model.add_loss([loss])
		self.output_model = Model([src_seq_input, tgt_seq_input], final_output)
		
		self.model.compile(optimizer, None)
		self.model.metrics_names.append('ppl')
		self.model.metrics_tensors.append(self.ppl)
		self.model.metrics_names.append('accu')
		self.model.metrics_tensors.append(self.accu)

	def make_src_seq_matrix(self, input_seq):
		src_seq = np.zeros((1, len(input_seq)+3), dtype='int32')
		src_seq[0,0] = self.i_tokens.startid()
		for i, z in enumerate(input_seq): src_seq[0,1+i] = self.i_tokens.id(z)
		src_seq[0,len(input_seq)+1] = self.i_tokens.endid()
		return src_seq

	def decode_sequence(self, input_seq, delimiter=''):
		src_seq = self.make_src_seq_matrix(input_seq)
		decoded_tokens = []
		target_seq = np.zeros((1, self.len_limit), dtype='int32')
		target_seq[0,0] = self.o_tokens.startid()
		for i in range(self.len_limit-1):
			output = self.output_model.predict_on_batch([src_seq, target_seq])
			sampled_index = np.argmax(output[0,i,:])
			sampled_token = self.o_tokens.token(sampled_index)
			decoded_tokens.append(sampled_token)
			if sampled_index == self.o_tokens.endid(): break
			target_seq[0,i+1] = sampled_index
		return delimiter.join(decoded_tokens[:-1])

	def make_fast_decode_model(self):
		src_seq_input = Input(shape=(None,), dtype='int32')
		tgt_seq_input = Input(shape=(None,), dtype='int32')
		src_seq = src_seq_input
		tgt_seq = tgt_seq_input

		src_pos = Lambda(self.get_pos_seq)(src_seq)
		tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
		if not self.src_loc_info: src_pos = None
		enc_output = self.encoder(src_seq, src_pos)
		self.encode_model = Model(src_seq_input, enc_output)

		enc_ret_input = Input(shape=(None, self.d_model))
		dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_ret_input)	
		final_output = self.target_layer(dec_output)
		self.decode_model = Model([src_seq_input, enc_ret_input, tgt_seq_input], final_output)
		
		self.encode_model.compile('adam', 'mse')
		self.decode_model.compile('adam', 'mse')

	def decode_sequence_fast(self, input_seq, delimiter=''):
		if self.decode_model is None: self.make_fast_decode_model()
		src_seq = self.make_src_seq_matrix(input_seq)
		enc_ret = self.encode_model.predict_on_batch(src_seq)

		decoded_tokens = []
		target_seq = np.zeros((1, self.len_limit), dtype='int32')
		target_seq[0,0] = self.o_tokens.startid()
		for i in range(self.len_limit-1):
			output = self.decode_model.predict_on_batch([src_seq,enc_ret,target_seq])
			sampled_index = np.argmax(output[0,i,:])
			sampled_token = self.o_tokens.token(sampled_index)
			decoded_tokens.append(sampled_token)
			if sampled_index == self.o_tokens.endid(): break
			target_seq[0,i+1] = sampled_index
		return delimiter.join(decoded_tokens[:-1])

	def beam_search(self, input_seq, topk=5, delimiter=''):
		if self.decode_model is None: self.make_fast_decode_model()
		src_seq = self.make_src_seq_matrix(input_seq)
		src_seq = src_seq.repeat(topk, 0)
		enc_ret = self.encode_model.predict_on_batch(src_seq)

		final_results = []
		decoded_tokens = [[] for _ in range(topk)]
		decoded_logps = [0] * topk
		lastk = 1
		target_seq = np.zeros((topk, self.len_limit), dtype='int32')
		target_seq[:,0] = self.o_tokens.startid()
		for i in range(self.len_limit-1):
			if lastk == 0 or len(final_results) > topk * 3: break
			output = self.decode_model.predict_on_batch([src_seq,enc_ret,target_seq])
			output = np.exp(output[:,i,:])
			output = np.log(output / np.sum(output, -1, keepdims=True) + 1e-8)
			cands = []
			for k, wprobs in zip(range(lastk), output):
				if target_seq[k,i] == self.o_tokens.endid(): continue
				wsorted = sorted(list(enumerate(wprobs)), key=lambda x:x[-1], reverse=True)
				for wid, wp in wsorted[:topk]: 
					cands.append( (k, wid, decoded_logps[k]+wp) )
			cands.sort(key=lambda x:x[-1], reverse=True)
			cands = cands[:topk]
			backup_seq = target_seq.copy()
			for kk, zz in enumerate(cands):
				k, wid, wprob = zz
				target_seq[kk,] = backup_seq[k]
				target_seq[kk,i+1] = wid
				decoded_logps[kk] = wprob
				decoded_tokens.append(decoded_tokens[k] + [self.o_tokens.token(wid)]) 
				if wid == self.o_tokens.endid(): final_results.append( (decoded_tokens[k], wprob) )
			decoded_tokens = decoded_tokens[topk:]
			lastk = len(cands)
		final_results = [(x,y/(len(x)+1)) for x,y in final_results]
		final_results.sort(key=lambda x:x[-1], reverse=True)
		final_results = [(delimiter.join(x),y) for x,y in final_results]
		return final_results



```


#### 参考

[放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)

[从Seq2seq到Attention模型到Self Attention（二）](https://wallstreetcn.com/articles/3417279)

[NLP学习笔记（一）：图解Transformer+实战](https://blog.csdn.net/linxid/article/details/84321617)
