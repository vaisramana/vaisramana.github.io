

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                    tex2jax: {
                    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
                    inlineMath: [['$','$']]
                    }
                });
    </script>
</head>


# 解析Transformer模型
> https://mp.weixin.qq.com/s/eQ64XUkTxyxjYmkciFUofQ
> Tensorflow官方transformer实现
> https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb

## Scaled-Dot Attention
```python
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights
```

输入的是Q, K, V矩阵和一个mask掩码向量，根据公式进行矩阵相乘，得到最终的输出，以及注意力权重。
- 输入两个矩阵QK尺寸分别是$[seq_q, depth]$和$[seq_k, depth]$，
	- 如果是单头attention，那么QKV只有一组，那么$depth==1$
	- 如果$depth>1$的话属于多头attention。
- 两者矩阵乘$Q\times K^T$得到$[seq_q, seq_k]$
	- 如果是单头attention，那么$[seq_q, 1]\times [1, seq_k] -> [seq_q, seq_k]$，输出矩阵中$[i,j]$元素表示，第$i$个元素和第$j$个元素之间的相关度
	- 如果是多头attention，那么$[seq_q, depth]\times [depth, seq_k] -> [seq_q, seq_k]$，输出矩阵中$[i,j]$元素表示，在累加所有$depth$组QK之后，第$i$个元素和第$j$个元素之间的相关度
- 除以$\sqrt{seq_k}$做一次scale
- 如果输入有mask，加上mask
- 对于矩阵$[seq_q, seq_k]$在$seq_k$维度做softmax，得到权重，也就是第$i$个元素和第$j$个元素之间的归一化的相关度
- 输入矩阵V尺寸是$[seq_v, depth_v]$，表示原始输入被编码成$depth_v$个新特征，每个特征序列长度是$seq_v$。因为要做矩阵乘法，所有必须有$seq_v==seq_k$，$[seq_q, seq_k]\times [seq_v, depth_v] -> [seq_q, depth_v]$

对于输出的$[seq_q, depth_v]$矩阵，相当于原始输入被编码成$depth_v$个新特征，每个特征序列长度是$seq_v=seq_k$，再经过一次各个特征之间的加权，








# 从发展历史视角解析Transformer
> https://mp.weixin.qq.com/s/h1fdIHQiPt6MfaaQjKe9zQ
> the transformer … “explained”?
> https://nostalgebraist.tumblr.com/post/185326092369/the-transformer-explained


# 经典的全连接神经网络
每个不同的输入变量都是独一无二的雪花算法（snowflake）。当全连接神经网络学会识别涉及某个特定变量或其集合时，它们不会对其他变量或其集合进行任何自动泛化。
但如果输入变量包含某种已知的、结构化的关系，比如空间或时间布局，全连接神经网络的表现就会很差。
如果输入是图片中的像素，那么全连接网络就无法学习类似“左侧的像素较亮，右侧的像素较暗”的模式，而是必须分别学习“(0, 0) 比 (1, 0) 亮”，“(1, 0) 比 (2, 0) 亮”以及“ (0, 1) 比 (1, 1) 亮”等等。


# 卷积神经网络
卷积神经网络（CNN）了解输入的空间布局，并用相对的术语处理输入：CNN不学习“在位置（572，35）处的像素”，而是学习“我正看着的中心的像素”、“左边的像素”等等。然后，它们一边“看”向图像的不同部分一边滑动，在每个区域中寻找相对中心的相同图案。

图像和文本的区别：特征是否可以独立理解
- 图像中的每一个突出事物（狗，狗的鼻子，边缘，一小块颜色）都可以分开来独立理解，不需要观察该事物以外的内容。除非是在一些奇奇怪怪的场景，否则通常不会出现“哦，我现在看到一只狗，但我必须观察狗以外的事物才确认这是一只狗”的情况。所以，你可以从一些小细节入手，然后分层次深入思考：“啊，这是边缘部分–>啊，那是一个由边缘组成的长方形物体–>啊，那是狗的鼻子–>啊，那是狗的头部–>啊，那是一只狗。”物体的每个部位都是由它所包含的更小的特征来定义的。
- 文本处理中句子中的代词可能出现在句首，但是指代的先行词常常在句末。我们没有办法在不改变句意的前提下，将一个句子准确无误地分解成可以独立理解的分句，然后再相互链接。所以CNN的局部性原理不利于文本处理。

# 循环神经网络
RNN并非观察当前位置以及位置周围的局部小窗口，而是观察下列信息：
- 当前位置
- 观察最后位置之后的输出

当输入是文本格式时，感觉就像“阅读”：RNN处理第一个单词，概括当时所收集到的所有信息；然后根据概括的信息处理第二个单词，更新概括的内容；再根据新的概括内容处理第三个单词，再次更新概括信息，循环往复。

RNN面临的问题
- 单一方向
RNN每次只能沿一个方向“读取”，这就造成了不对称的问题：在句首附近，输出只能使用几个单词所构成的信息；在句尾附近，输出则可以使用所有单词构成的信息。这一点与CNN相反，因为CNN对每个位置的信息的处理方式都是一样的。
在这种情况下，如果一个句子开头的单词只能根据后面出现的单词来理解时，就会出现问题。RNN可以基于前面的单词理解后面的单词（这也是RNN的核心观点），但不能基于后面的单词来理解前面的单词。
这个问题可以通过两种方式在一定程度上进行回避：一是使用多个RNN层，其中较新的层类似“附加阅读通道”；二是用两个RNN分别从不同方向读取（这也是“BiLSTMs”的基本原理）。
- 长度有限的"scratchpad”（便签存储器）
RNN只能使用长度有限的"scratchpad”（便签存储器）来处理单词之间的依赖关系，并且必须使用同一个“scratchpad”来处理所有短期和长期依赖关系。


# 注意力机制
在作者看来，注意力机制的提出最初是为了处理成对的文本，比如将句子1与句子2中的每个词/短语进行比较，以找出哪些词/短语可能是指同一个主题或其他。
注意力机制是为了比较两种不同的文本而提出的。但注意力机制也可以用来比较同一个文本。这被称为“自注意力机制”。

## 介绍一个注意力运行方式
- a key：某单词的简介，例如“bike”的简介可能包含“我是中性名词”
- a query：某单词在查阅简介时，搜寻的是什么信息？例如“it”这类代词可能是：“我匹配的是中性名词”。
- a value：某单词含义的其他信息，可能与匹配过程无关例如，有关“bike”含义的其他信息

对于每个单词，你可以利用key和query来确定该单词与自己本身的匹配度，以及与其他单词的匹配度。然后，你可以汇总价值信息，用“匹配分数”（match scores）进行加权。最后，你可能会得到一个既包含原单词的大多数价值、又包含其他单词的些许价值的结果，比如“我仍然是一个代词，但同时我还指代了这个名词，且表达的就是该名词的含义。”

“multi-headed” attention，一个单词内包含多少条key/query/value






