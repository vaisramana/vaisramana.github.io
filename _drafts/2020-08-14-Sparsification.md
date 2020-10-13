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



# Sparse matrix
http://www.bu.edu/pasi/files/2011/01/NathanBell1-10-1000.pdf
> chinaXiv:201611.00855v1

有稀疏矩阵A
$$
A = \begin{pmatrix}
1 & 5 & 0 & 0 \\
0 & 2 & 6 & 0 \\
8 & 0 & 3 & 7 \\
0 & 9 & 0 & 4 \\
\end{pmatrix} \\
$$
考虑稀疏矩阵向量乘法SpMV，一个稀疏矩阵乘以一个普通向量得到一个向量
$$
SPM_{M\times N} \times X_{N\times 1} = Y_{M\times 1}
$$

## COO
Coordinate（COO）
每个元素用三元组来表示，（行号，列号，数值），每个三元组都可以凭借自己的信息定位，但相比于CSR更费空间
$$
data = \begin{bmatrix}
1 & 5 & 2 & 6 & 8 & 3 & 7 & 9 & 4 \\
\end{bmatrix} \\
row = \begin{bmatrix}
0 & 0 & 1 & 1 & 2 & 2 & 2 & 3 & 3 \\
\end{bmatrix} \\
col = \begin{bmatrix}
0 & 1 & 1 & 2 & 0 & 2 & 3 & 1 & 3 \\
\end{bmatrix} \\
$$

其中x是输入向量，y是输出向量
```c++
for(i=0; i<num_nonzeros; i++)
{
	y[rows[i]] += data[i]*x[cols[i]];
}  
```

## DOK
Dictionary of keys（DOK）
DOK的存储格式与COO格式相同，只是用字典变量存数稀疏矩阵的矩阵元。行列值作为字典的键，矩阵元作为字典内容。

## CSR/CRS/Yale format
compressed sparse row (CSR)，compressed row storage (CRS)
**适合内存按行排布的矩阵**
矩阵$M$由三个一维向量表示，分别是$V, INDEX_{COL}, INDEX_{ROW}$，由$NNZ$表示矩阵$M$中的非零元素个数，其中
- $V, INDEX_{COL}$向量长度都是$NNZ$
- $INDEX_{ROW}$向量长度是矩阵$M$的行数+1，用来指示$V$中的元素属于哪一行，第一个元素永远是0

$$
M = \begin{pmatrix}
0 & 0 & 0 & 0 \\
5 & 8 & 0 & 0 \\
0 & 0 & 3 & 0 \\
0 & 6 & 0 & 0 \\
\end{pmatrix} \\
V = \begin{bmatrix} 5 & 8 & 3 & 6 \end{bmatrix} \\
INDEX_{COL} = \begin{bmatrix} 0 & 1 & 2 & 1 \\ \end{bmatrix} \\
INDEX_{ROW} = \begin{bmatrix} 0 & 0 & 2 & 3 & 4 \\ \end{bmatrix} \\
$$

用
```c++
row_start = ROW_INDEX[row_idx]
row_end   = ROW_INDEX[row_idx + 1]
```
来表示矩阵$M$第row_idx行的元素对应$V$中的元素index，比如例子中，
- 第一行没有非零元素，所以row_start=0，row_end=0
- 第二行有两个非零元素，所以row_start=0，row_end=2，从$V$中取出index 0和1元素对应非零值，从$INDEX_{COL}$取出index 0和1元素对应列号。
- 第三行有一个非零元素，所以row_start=2，row_end=3，从$V$中取出index 2元素对应非零值，从$INDEX_{COL}$取出index 2元素对应列号。

$$
data = \begin{bmatrix}
1 & 5 & 2 & 6 & 8 & 3 & 7 & 9 & 4 \\
\end{bmatrix} \\
row = \begin{bmatrix}
0 & 2 & 4 & 7 & 9 \\
\end{bmatrix} \\
col = \begin{bmatrix}
0 & 1 & 1 & 2 & 0 & 2 & 3 & 1 & 3 \\
\end{bmatrix} \\
$$

```c++
for(i=0;i<num_rows;i++){
	start=row[i];
	end =row[i+1];
	sum =y[i];
	for(jj=start;jj< end;jj++)
		sum += x[col[jj]]* data[jj];
	y[i]=sum;
} 
```


## CSC
compressed sparse column (CSC)，compressed column storage (CCS)
**适合内存按行排布的矩阵**

## BSR
Block Sparse Row
分块稀疏矩阵，结构化稀疏矩阵


## DIA
Diagonal (DIA)
对角线存储法，按对角线方式存，列代表对角线，行代表行。省略全零的对角线。(从左下往右上开始：第一个对角线是零忽略，第二个对角线是8，9，第三个对角线是零忽略，第四个对角线是1，2，3，4，第五个对角线是5，6，7，第六第七个对角线忽略)。
如果原始矩阵就是一个对角性很好的矩阵那压缩率会非常高
$$
data = \begin{bmatrix}
* & 1 & 5 \\
* & 2 & 6 \\
8 & 3 & 7 \\
9 & 4 & * \\
\end{bmatrix} \\
offset = \begin{bmatrix}
-2 & 0 & 1 \\
\end{bmatrix} \\
$$

```c++
for(i=0; i< num_diags; i++) {
	k = offsets[i]; //diagonal offset
	Istart = max((0,-k);
	Jstart = max(0, k);
	N = min (num_rows - Istart, num_cols - Jstart);
	for(n=0; n<N; n++) {
 		y_[Istart+n] += data[Istart+i*stride+n]*x[Jstart+ n];
	}
} 
```


## ELL
ELLPACK (ELL)
用两个和原始矩阵相同行数的矩阵来存：第一个矩阵存的是列号，第二个矩阵存的是数值，行号就不存了，用自身所在的行来表示；这两个矩阵每一行都是从头开始放，如果没有元素了就用个标志比如*结束。
$$
indices = \begin{bmatrix}
0 & 1 & * \\
1 & 2 & * \\
0 & 2 & 3 \\
1 & 3 & * \\
\end{bmatrix} \\
data = \begin{bmatrix}
1 & 5 & * \\
2 & 6 & * \\
8 & 3 & 7 \\
9 & 4 & * \\
\end{bmatrix} \\
$$

```c++
for(n=0; n< max_ncols; n++)
{
	for(i=0; i<num_rows; i++)
		y[i] += data[n*num_rows+i] * x[indices[n*num_rows+i]];
} 
```

## summary

| 存储格式 | X 访存特征 | 内层循环长度 | 额外计算             | 写 Y 次数      |
| -------- | ---------- | ------------ | -------------------- | -------------- |
| DIA      | 连续       | 基本相等     | 由非零元所占比例决定 | 矩阵对角线条数 |
| ELL      | 不规则     | 相等         | 由非零元所占比例决定 | 该矩阵最大行度 |
| CSR      | 不规则     | 基本不等     | 无                   |                |
| COO      | 不规则     | --           | 无                   |                |

- DIA
当对角线条数不多，且矩阵总体非零元所占比例较大或“真对角线”较多时，相对 CSR格式 DIA 格式在稀疏矩阵向量乘法程序上表现出较明显的性能优势。
可能引入不必要的零元素填充。
- ELL
当稀疏矩阵的最大行度较小，每行行度波动较小且非零元所占比例较大时， ELL 稀疏矩阵向量乘法相对 CSR 格式有性能优势。
可能引入不必要的零元素填充。
- CSR/COO
虽然在 GPU 上使用 COO 格式稀疏矩阵向量乘法在幂律矩阵上取得了四种格式中的最好性能，但在 CPU 上使用并没有相同的表现。



# Compression of Deep Learning Models for Text: A Survey
> Microsoft, India
> https://arxiv.org/abs/2008.05221



# Efficient Neural Audio Synthesis
> google
> ICML 2018
> https://arxiv.org/abs/1802.08435





# Comparing Rewinding and Fine-tuning in Neural Network Pruning
> https://arxiv.org/abs/1912.05671
# Linear Mode Connectivity and the Lottery Ticket Hypothesis
> MIT
> ICLR 2020
> https://arxiv.org/abs/2003.02389
> https://github.com/lottery-ticket/rewinding-iclr20-public

对比rewinding和fine-tuning技术


# To prune, or not to prune: exploring the efficacy of pruning for model compression
> Stanford & Google
> 2017 NIPS
> https://arxiv.org/abs/1710.01878
> https://github.com/tensorflow/tensorflow/tree/v1.15.0-rc0/tensorflow/contrib/model_pruning



# The State of Sparsity in Deep Neural Networks
> Google
> ICML 2019
> https://arxiv.org/abs/1902.09574
> https://github.com/LONG-9621/Stackedcapsule/tree/master/state_of_sparsity



# Recurrent Neural Network Regularization
> Under review as a conference paper at ICLR 2015
> https://arxiv.org/abs/1409.2329
> https://github.com/wojzaremba/lstm

## 3 REGULARIZING RNNS WITH LSTM CELLS
### 3.1 LONG-SHORT TERM MEMORY UNITS
描述原始LSTM

$$
LSTM: h^{l-1}_{t},h^{l}_{t-1},c^{l}_{t-1} \to h^{l}_{t},c^{l}_{t} \\
\begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} = 
\begin{pmatrix} sigm \\ sigm \\ sigm \\ tanh \end{pmatrix}
T_{2n,4n} 
\begin{pmatrix} h^{l-1}_{t} \\ h^{l}_{t-1} \end{pmatrix} \\
c^{l}_{t}=f \odot c^{l}_{t-1} + i \odot g \\
h^{l}_{t}=o \odot tanh(c^{l}_{t})
$$

其中$h^{l}_{t}$表示$t$时刻第$l$层的状态输出，那么对于当前第$l$层来说，$h^{l-1}_{t}$就是第$l$层的输入。$T_{2n,4n} $表示一个线性变换。


### 3.2 REGULARIZATION WITH DROPOUT
本文贡献在于在LSTM上应用dropout，成功减少过拟合。主要的想法是只在非循环链接上应用dropout，即只在输入上应用dropout。
$$
\begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} = 
\begin{pmatrix} sigm \\ sigm \\ sigm \\ tanh \end{pmatrix}
T_{2n,4n} 
\begin{pmatrix} {\bf D}(h^{l-1}_{t}) \\ h^{l}_{t-1} \end{pmatrix} \\
c^{l}_{t}=f \odot c^{l}_{t-1} + i \odot g \\
h^{l}_{t}=o \odot tanh(c^{l}_{t})
$$

其中${\bf D}$表示dropout

对于GRU
$$
\begin{pmatrix} r \\ z \\ n \end{pmatrix} = 
\begin{pmatrix} sigm \\ sigm \\ tanh \end{pmatrix}
T_{2n,3n} 
\begin{pmatrix} x_t \\ h_{t-1} \end{pmatrix} \\
h_{t}=(1-z) \odot n_{t} + z_t \odot h_{t-1}
$$



# Learning Structured Sparsity in Deep Neural Networks
> Accepted by NIPS 2016
> University of Pittsburgh
> https://arxiv.org/abs/1608.03665


# Learning Intrinsic Sparse Structures within Long Short-Term Memory
> Published in ICLR 2018
> Duke University & Business AI and Bing Microsoft
> https://arxiv.org/abs/1709.05027
> https://github.com/wenwei202/iss-rnns

## 1 INTRODUCTION
非结构化的稀疏对于硬件系统并不友好，加速效果非常差，GRU上alexnet的稀疏率分别是67.6%, 92.4%, 97.2%, 96.6%和94.3%时，加速分别为0.25×, 0.52×, 1.38×, 1.04×和1.36×。CPU上也类似，稀疏率达到80%以上才有增益，95%有3-4倍增益，而95%的理论值是20倍。

## 3 LEARNING INTRINSIC SPARSE STRUCTURES
### 3.1 INTRINSIC SPARSE STRUCTURES
LSTM实现
$$
i_t=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_{i}) \\
f_t=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_{f}) \\
o_t=\sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_{o}) \\
u_t=tanh(W_{xu}x_t+W_{hu}h_{t-1}+b_{u}) \\
c_t=f_t\odot c_{t_1}+i_t\odot g_t \\
h_t=o_t\odot tanh(c_t)
$$

实现中将8个权重矩阵拼接成一个大矩阵，将输入$x$和$h_{t-1}$ concat起来作为一个整体，输入这个大矩阵。






# Skip RNN: learning to skip state updates in recurrent neural networks
> Accepted as conference paper at ICLR 2018
> Google
> https://arxiv.org/abs/1708.06834
> https://imatge-upc.github.io/skiprnn-2017-telecombcn/



# TinyLSTMs: Efficient Neural Speech Enhancement for Hearing Aids
> Arm ML Research Lab
> [v1] Wed, 20 May 2020
> https://arxiv.org/pdf/2005.11138.pdf
> https://github.com/BoseCorp/efficient-neural-speech-enhancement

## 3. Optimizing LSTMs for HA Hardware
### 3.1. Structural Pruning
分为结构化pruning和随机pruning，随机pruning除非稀疏率非常高，在真实硬件上很难部署。结构化pruning将权重$\theta$分为若干个group $\Gamma$，每个group内包含一组权重$w_g \in \Gamma$。
对于第$k$层LSTM的第$g$组权重，增加一个binary mask $r_g$
$$
r_g=\Bbb{l}(|| w_g ||_2 - \tau_k \geq 0)
$$
其中$\tau_k$是可训练参数，$\Bbb{l}$是指示函数，即
$$
r_g=\Bbb{l}(|| w_g ||_2 - \tau_k \geq 0)=
\begin{cases}
1,  & \text{if } || w_g ||_2 - \tau_k \geq 0 \\
0, & \text{else}
\end{cases}
$$
为了保证反向传播，指示函数$\Bbb{l}$会选择sigmoid。
**意义为，让平方值比$\tau_k$小的权重值直接变成0，$\tau_k$越大稀疏率越高。**
让全图的mask为$P=\{r_g, 1 \leq g \leq |\Gamma|\}$，mask之后的权重为$\theta \bigodot P$。在loss目标函数里加上权重平方惩罚
$$
\lambda \sum_{g=1}^{|\Gamma|}r_g|| w_g ||_2
$$

其中$\lambda$是超参数，控制Pruning程度。
>Learning Intrinsic Sparse Structures within Long Short-Term Memory

中手动选择阈值$\tau_k$，而本文中变为可训练。

### 3.2. Quantization
$w \in R$表示浮点值，$Q_{\alpha,\beta}(w)$对应其定点值，量化操作表示在动态范围$(\alpha,\beta)$内用均匀量化到$2^{bits}-1$个定点值。
$$
Q_{\alpha,\beta}(w)=s*round((chip(w,\alpha,\beta)-\alpha)/s)+\alpha \\
s=(2^{bits}-1)/(\beta-\alpha)
$$
这里用了google的方法
> Quantization and training of neural networks for efficient integer-arithmetic-only inference

权重和激活函数都量化到8bit，但是mask本身量化到16bits。

### 3.3. Skip RNN Cells
> Skip RNN: learning to skip state updates in recurrent neural networks

是基于RNN的skip，本文中基于LSTM的skip

Skip RNN Cells可以被看做是动态的临时pruning，引入一个一个binary 神经元$g^t \in {0,1}$，称为state update gate，用于选择是否RNN state会被更新或者维持，原文中作用于RNN状态$s$，这里作用于LSTM状态$c$和$h$
LSTM流程为
$$
i^t=\sigma(W_{xi}x^t+W_{hi}h^{t-1}+b_{i}) \\
r^t=\sigma(W_{xr}x^t+W_{hr}h^{t-1}+b_{r}) \\
o^t=\sigma(W_{xo}x^t+W_{ho}h^{t-1}+b_{o}) \\
u^t=tanh(W_{xu}x^t+W_{hu}h^{t-1}+b_{u}) \\
c^t=r_t\odot c^{t_1}+i^t\odot g^t \\
h^t=o^t\odot tanh(c^t)
$$
以RNN为例
$$
g_t=f_{binarize}(\hat{g}^t)=round(\hat{g}^t) \\
s^t=g^t \hat{s}^t+(1-g)s^{t-1}
$$
$\hat{g}^t$是概率值，$\hat{g}^t \in [0,1]$，实现中是一个四舍五入round函数
因为$g^t \in {0,1}$，那么$s^t$要么是当前产生的新值$\hat{s}^t$，要么是上一帧的状态$s^{t-1}$。
**参数量不会改变，但是现实中flop是根据输入不同而动态变化的，有些帧不需要更新状态，因此也就没有计算量。**

## 4. Experimental Results
最终评估用signal-to-distortion ratio (SDR)，训练中用scale-invariant signal-to-distortion ratio (SISDR)，SISDR计算量更少而且和SDR相关性很好。
> SDR https://arxiv.org/pdf/1603.04179.pdf
> SISDR https://arxiv.org/abs/1811.02508

经验
- 结构化pruning可以达到47%模型大小而不影响性能
- 同时结构化pruning和量化，达到37%模型大小，SISDR下降0.2db















