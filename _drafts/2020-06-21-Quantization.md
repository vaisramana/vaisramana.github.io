

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




# Compression of Deep Learning Models for Text: A Survey
> Microsoft, India
> https://arxiv.org/abs/2008.05221





# EasyQuant: Post-training Quantization via Scale Optimization
> DeepGlint & OPEN AI LAB
> https://arxiv.org/abs/2006.16669

## 1. Introduction
相比于post-training量化，基于训练的量化有更高的准确性，但缺点是
- 直接训练量化模型很耗时
- 需要专家知识指导训练过程，目标任务领域和量化领域
- 某些场景下，量化过程无法获取所有完整训练数据

这里提出EasyQuant (EQ)
- 提出一种scale优化方法，将量化卷积过程看做一个优化问题，最大化FP32和INT8输出之间的余弦相似度。通过交替搜索权重和激活函数scale来解决这个优化问题。整个网络层面，逐层优化scale，权重和激活函数scale联合优化，下一层优化基于上一层量化参数。
- 实现INT7量化推理，提高INT16中间变量的使用效率
- 在不同CV任务中，这种scale搜索方法可以达到高效INT8量化，并且INT7量化也可以接近INT8性能。在ARM平台上实现了EQ INT7推理。


## 2. Related Work
### 2.1 Training-based Quantization
### 2.2 Post-training Quantization
比较有代表的是英伟达的TensorRT (TRT)和google的TFLite
- TRT用KL散度或者相对熵最小值来标定激活函数的量化threshold，用最大绝对值来标定权重量化threshold
- TFLite用最大绝对值来标定激活函数和权重，权重还用了per-channel方法的量化


## 3. The Proposed Method
### 3.1 Linear Quantization Formulation
$X$是浮点输入，$S$是正实数scale，量化之后$Q(X, S)\in Z_b$，$Z_b$表示$b$ bit位宽的整形数。线性量化包含三个步骤
- scale
- round
scale之后的tensor需要ceiling rounding到整形
- clipped
如果tensor超过动态范围

$$
Q(X, S)=Clip(Round(X\cdot S))
$$

定义一个量化的$L$层网络为$\{A_l, W_l, S_l\}^{L}_{l=1}$
其中$A_l, W_l, S_l$分别是第$l$层激活函数，权重和量化scale。
量化scale $S_l$包含两部分，激活函数scale $S^a_l$，权重scale $S^w_l$，
假设第$l$层输出为$O_l$，量化格式为$\hat O_l$，那么有

$$
\hat O_l=\frac{Q(A_l, S^a_l)*Q(W_l, S^w_l)}{S^a_l\cdot S^w_l} \\
O_l=A_l*W_l
$$

这里*代表卷积，$A_l$实际是输入？
目标函数用于优化scale $S^a_l$和$S^w_l$来增加$\hat O_l$和$O_l$的相似性。



# Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks
> IEEE ICCV 2019
> https://arxiv.org/abs/1908.05033

Differentiable Soft Quantization (DSQ)用双曲正切函数来代替低比特量化的楼梯函数，保证可微分。

ARM平台部署时，MLA指令会将两个8bit寄存器相乘并累加到另一个bit寄存器中，考虑到保存累加的寄存器可能溢出，会用SADDW指令再转换到16bit寄存器中。
SADDW会有额外的计算开销，需要尽可能减少SADDW的次数，考虑b-bit带符号数的量化乘加操作，两个b-bit带符号数乘法的最大值是$(-2^b-1)^2$，那么8-bit的寄存器只能保证$\frac{2^7-1}{(-2^b-1)^2}$次运算不溢出，可见量化bit数越小，节省的SADDW次数越多

| b         | 2    | 3    | 4    |
| --------- | ---- | ---- | ---- |
| MLA/SADDW | 31/1 | 7/1  | 1/1  |

可见4bit量化每次MLA结果就必须用16bit存储，而2bit量化可以做到累加31次MLA才转换到16bit。
