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


Do RNN and LSTM have Long Memory?

>https://arxiv.org/abs/2006.03860
>https://mp.weixin.qq.com/s/rZbQESxa972QQbU5jrP_jg
>https://github.com/Gladys-Zhao/mRNN-mLSTM/tree/56ee764cf50d4d84c606b9c6b98107907bb2beca



# 2. Memory Property of Recurrent Networks
## 2.1. Background
统计学中长期记忆的定义
$\{X_t,t \in Z\}$是一个平稳单变量过程，自协方差autocovariance函数为$\gamma_x(k)$，其中$k\in Z$，谱密度函数为$f_x(\lambda)=\frac{1}{2\pi}\sum_{k=-\infty}^{\infty}{\gamma_x(k)e^{-ijk}}$，其中$\lambda\in [-\pi,\pi]$，那么

- 如果$\sum_{k=-\infty}^{\infty}{\gamma_x(k)}=\infty$，那么$\{X_t\}$就是有长期记忆的
- 如果$0<\sum_{k=-\infty}^{\infty}{\gamma_x(k)}<\infty$，那么$\{X_t\}$就是有短期记忆的

**简单说长期记忆过程的自协方差不可和，累积无穷大，或者谱密度无穷大，短期记忆过程的自协方差可和，谱密度有上限值。**

一个典型的长期记忆操作是fractionally integrated process，假设$B$为后移运算符，对于$j>=0$和一随机变量时间序列$X_t$，有$B^jX_t=X_{t-j}$，比如
$$
(1-B)^2X_t=(1-2B+B^2)X_t=X_t-2BX_t+B^2X_t=X_t-2X_{t-1}+X_{t-2}
$$
fractionally integrated process定义为
$$
(1-B)^dY_t=X_t \\
(1-B)^d=\sum_{j=0}^{\infty}\frac{\Gamma(d+j)}{j!\Gamma(d)}B^j=:\sum_{j=0}^{\infty}w_j(d)B^j
$$
其中$\Gamma$是Gamma函数，一般$X_t$会被选取为ARMA自回归滑动平均模型Autoregressive moving-average model，具有短时记忆，那么$Y_t$就服从ARFIMA分数差分整合移动平均自回归模型Autoregressive fractionally-integrated moving-average model。
可以证明
$$
w_j(d) \sim j^{-d-1}
$$
当$j\to \infty \quad k\to \infty $时，自协方差autocovariance函数为$\gamma_x(k)\to |k|^{2d-1}$。
- $d<=0$时，$\gamma_x(k)\to |k|^{2d-1}=|k|^{-1} \to 0$
- $d\in (0,0.5)$时，$\gamma_x(k)\to |k|^{2d-1} \in (|k|^{-1}, 1)$，具有短时记忆
- $d> 0.5$时，$\gamma_x(k)\to |k|^{2d-1} > 1$，具有长时记忆

ARFIMA中ARMA部分负责对短时记忆建模，而分数差分部分负责对长时记忆建模。

ARFIMA特性
- $d\in (-0.5,0.5)$时模型平稳，$d> 0.5$时模型非平稳
- 对于平稳模型，$d\in (-0.5,0)$时模型具有短时记忆，$d\in (0,0.5)$时模型具有长时记忆，$d$越大自协方差越大，长时记忆效果越长
- $Y_t$自协方差函数已多项式速率衰减$w_j(d) \sim j^{-d-1}$，而不是指数速率衰减，多项式速率衰减序列在指数小于-1时不可和无限大，而指数衰减序列总是可和的。

马尔科夫链在$n$步之后的条件分布$P^n{x,.}$，随着$n$增大会以指数速率收敛到平稳分布$\pi(.)$，意味着马尔科夫链在$x$状态的信息已指数速率丢失了，不具有长期记忆。

## 2.2. Recurrent Network Process
假设递归网络输入$\{x^{(t)}\}$，输出$\{z^{(t)}\}$，目标序列$\{y^{(t)}\}$来自数据生成过程，或者说模型
$$
$y^{(t)}=z^{(t)}+\epsilon^{(t)}
$$

$\epsilon^{(t)}$是一个iid独立一致分布噪声。这个额外的噪声对应loss函数，用于衡量$y^{(t)}$和$x^{(t)}$之间的距离。
我们目标是检查一个网络能否对长时记忆建模。
引入隐层状态$s^{(t)}$，递归网络可以重写成马尔科夫链形式
$$
\left( \begin{matrix} y^{(t)} \\ s^{(t)} \end{matrix} \right) =
M(y^{(t-1)},s^{(t-1)})+
\left( \begin{matrix} \epsilon^{(t)} \\ 0 \end{matrix} \right)
$$
考虑到网络里转换函数$M$都是线性的，用$W$表示转换矩阵，那么
$$
\left(\begin{matrix} y^{(t)} \\ s^{(t)}\end{matrix}\right) =
W \left(\begin{matrix} y^{(t-1)} \\ s^{(t-1)} \end{matrix} \right) +
\left( \begin{matrix} \epsilon^{(t)} \\ 0 \end{matrix} \right)
$$


首先考虑vanilla RNN
$$
\begin{cases}
l^{(t)}=\left \| y^{(t)}-z^{(t)} \right\| \\
z^{(t)}=g(W_{zh}h^{(t)}+b_z) \\
h^{(t)}=\sigma(W_{hh}h^{(t-1)}+W_{hy}y^{(t-1)}+b_h)
\end{cases}
$$

改写成马尔科夫链形式
$$
\left( \begin{matrix} y^{(t)} \\ h^{(t)} \end{matrix} \right) =
M_{RNN}(y^{(t-1)},h^{(t-1)})+
\left( \begin{matrix} \epsilon^{(t)} \\ 0 \end{matrix} \right)
$$
其中$h^{(t)}$就是RNN的状态$h^{(t)}$，转移函数$M_{RNN}$定义为
$$
M_{RNN}=
\left( \begin{matrix} 
g(W_{zh}\sigma(W_{hh}h^{(t-1)}+W_{hy}y^{(t-1)}+b_h)+b_z) \\
\sigma(W_{hh}h^{(t-1)}+W_{hy}y^{(t-1)}+b_h)
\end{matrix} \right)
$$

同理，LSTM也可以重写成


## 2.3. Memory Property of Recurrent Network Processes
- Assumption 1.
\epsilon^{(t)}的联合密度函数连续且处处为正。对于$k>=2$，$E\left \| \epsilon^{(t)} \right\|<\infty$
- Theorem 1.
在Assumption 1.条件下，如果存在实数$0<a<1$和$b$，使得$\|M(x)\|\leq a \|x\|+b $，那么这个RNN过程有几何遍历性，因此有短时记忆
- Theorem 2.
在Assumption 1.条件下，如果线性RNN过程有几何遍历性且谱半径$\rho(W)<1$，那么这个RNN过程有短时记忆

推理1&2比较抽象，进一步得到引理1&2
- Corollary 1.
假设输出和激活函数分别是$g(.)$和$\sigma(.)$是连续而有界的，那么在Assumption 1.条件下，这个RNN过程就有几何遍历性且短时记忆。
- Corollary 2.


## 2.4. Long Memory Network Process
提出了一种对于神经网络适用的新的长期记忆的定义。假设神经网络可以写成（或近似成）下列形式
$$
y^{(t)}=\sum_{k=0}^{\infty}A_k x^{(t-k)}+\epsilon^{(t)}
$$
如果系数矩阵存在一个维度以多项式速率衰减，则认为网络具有长期记忆。





# 3. Long Memory Recurrent Networks
根据上述理论成果，我们想对RNN和LSTM做出最小程度的修改，使其获得对长期相关性建模的能力。类似于ARFIMA模型中的结构，我们给RNN和LSTM在不同位置添加了一个长期记忆滤波器，分别得到记忆增强RNN（Memory-augmented RNN，简称MRNN模型）和记忆增强LSTM模型（Memory-augmented LSTM，简称MLSTM模型）。

## 3.1. Memory-augmented RNN (MRNN)
原始RNN实现
$$
\begin{cases}
l^{(t)}=\left \| y^{(t)}-z^{(t)} \right\| \\
z^{(t)}=g(W_{zh}h^{(t)}+b_z) \\
h^{(t)}=tanh(W_{hh}h^{(t-1)}+W_{hx}x^{(t)}+b_h)
\end{cases}
$$

MRNN实现
$$
\begin{cases}
l^{(t)}=\left \| y^{(t)}-z^{(t)} \right\| \\
z^{(t)}=g(W_{zh}h^{(t)}+W_{zm}m^{(t)}+b_z) \\
h^{(t)}=tanh(W_{hh}h^{(t-1)}+W_{hx}x^{(t)}+b_h) \\
F(x^{(t)};d)_i=\sum_{i=1}^{K}w_j(d_i^{(t)})x_i^{t-j+1} \\
w_j(d)=\Gamma(d+j)/[j!\Gamma(d)]=\prod_{i=0}^{j-1}(i-d)/(i+1) \\
d^{(t)}=\frac{1}{2}\sigma(W_d[d^{(t-1)},h^{(t-1)},m^{(t-1)},x^{(t)}]+b_d) \\
m^{(t)}=tanh(W_{m}[m^{(t-1)}, F(x^{(t)};d^{(t)})]+b_m)
\end{cases}
$$

增加一个门$m^{(t)}$，$K$是个超参数，历史长度为$K$个输入，经过滤波器$F(x^{(t)};d)_i$再经过一个线性层得到$m^{(t)}$
$$
\begin{cases}
F(x^{(t)};d)_i=\sum_{i=1}^{K}w_j(d_i^{(t)})x_i^{t-j+1} \\
w_j(d)=\Gamma(d+j)/[j!\Gamma(d)]=\prod_{i=0}^{j-1}(i-d)/(i+1) \\
\end{cases}
$$


### implemtation


## 3.2. Memory-augmented LSTM (MLSTM)

# 4. Experiments
我们做了三个实验，一是在具有长期记忆性质的数据集上验证新提出的模型的优势，二是验证新提出的模型在只具有短期记忆性质的数据集上表现不会劣化，三是探究长期记忆滤波长度这一超参数对模型表现的影响。


## 4.1. Long Memory Datasets
- ARFIMA series
由ARFIMA模型生成的序列，2000+1200+800
- Dow Jones Industrial Average (DJI)
yahoo财经得到的2000-2019年每日数据取log，2500+1500+1029
- Metro interstate traffic volume
明尼阿波利斯的地铁人流量数据，1400+200+259
- Tree ring
树的年轮宽度数据，2500+1000+850

我们比较了八个模型：
1. 原始RNN，lookback = 1；
2. 双轨RNN（类似MRNN），但是滤波器部分不加限制，有个自由权重；
3. Recurrent weighted average network (RWA)；
4. MRNNF，即记忆参数不随时间变化；
5. MRNN，即记忆参数随时间变化；
6. 原始LSTM；
7. MLSTMF，即记忆参数不随时间变化；
8. MLSTM，即记忆参数随时间变化。
由于这些网络的训练是非凸问题，我们用不同的种子初始化模型会学到不同的模型，所以我们使用了100个不同的种子，并报告时间序列预测任务的误差度量的均值，标准差，以及最小值。

结论
- 基本上 MRNN/MRNNF > LSTM > MLSTM/MLSTMF > RNN

## 4.2. Short Memory Dataset
模型生成的无长时记忆序列，MRNN/MRNNF/MLSTM/MLSTMF与RNN性能相当，但是误差的方差略大。

## 4.3. Model Parameter K
实验$K=25,50,75,100$
- 对于MRNN/MRNNF，K越大RMSE越小
- 对于MLSTM/MLSTMF，K越小RMSE越小，可能因为MLSTM模型较大难训练的缘故


## 4.4. Sentiment Analysis
两个测试集CMU-MOSI和paper reviews dataset
虽然MLSTM和MLSTMF在时间序列预测数据集上优势不明显，但是在这个自然语言处理的分类任务上，优势则很明显了。

















