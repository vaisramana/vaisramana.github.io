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

Simple Recurrent Units for Highly Parallelizable Recurrence
>https://arxiv.org/abs/1709.02755

# 3. Simple Recurrent Unit
LSTM实现
$$
f_t=\sigma(W_fx_t+U_fh_{t-1}+b_f) \\
i_t=\sigma(W_ix_t+U_ih_{t-1}+b_i) \\
i_o=\sigma(W_ox_t+U_oh_{t-1}+b_o) \\
\widetilde{c_t}=\sigma(W_cx_t+U_ch_{t-1}+b_c) \\
c_t=f_t\circ c_{t-1}+i_t\circ \widetilde{c_t} \\
h_t = o_t\circ \sigma(c_t)
$$

GRU实现
$$
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z) \\
r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r) \\
n_t=tanh(W_hx_t+r_t\circ (U_hh_{t-1}+b_h)) \\
h_t=(1-z_t)\circ n_t+z_t\circ h_{t-1}
$$

SRU实现
$$
f_t=\sigma(W_{f}x_t+V_{f}\circ c_{t-1}+b_{f}) \\
c_t=(1-f_t)\circ(Wx_t)+f_t\circ c_{t-1} \\
r_t=\sigma(W_{r}x_t+V_{r}\circ c_{t-1}+b_{r}) \\
h_t=r_t\circ c_t + (1-r_t)\circ x_t
$$

SRU的两点改进：
- 矩阵乘法改为矩阵逐元素乘，$Vh_{t-1} \rightarrow V\circ h_{t-1}$
在GRU和LSTM中，$h_{t-1}$与权重矩阵做矩阵乘法得到各种门系数，但是$U_fh_{t-1}$这样的矩阵乘法很难并行，**各种门系数的的每一个维度都依赖于整个$h_{t-1}$，只有等整个$h_{t-1}$被算出来之后，才能得到门系数进而进一步计算。**
在SRU中，一个重要的改进是，**将矩阵乘法改为矩阵按位乘**。$h_t$计算出一行或者一列，就可以并行计算这一行或者一列对应的门系数。
- highway
GRU的输出中体现输入的部分是$(1-z_t)\circ n_t=(1-z_t)\circ tanh(W_hx_t+r_t\circ (U_hh_{t-1}+b_h))$，而SRU的输出中可以直接体现输入$(1-r_t)\circ x_t$，让梯度可以直接传到到上一层，体现了highway的思想。


加速主要来自于矩阵乘法改为矩阵逐元素乘，在隐藏层节点数$d$不变条件下
- $V$从$d\times d$变为$d$，模型参数量和计算量都变小
- 在维度$d$展开并行计算



## 3.1 Parallelized Implementation
针对CUDA的两点实现优化。
首先将输入序列$\{x_1, x_2, ..., x_L\}$的乘法batch起来， batched multiplication
$$
U^T=
\left(\begin{matrix}
   W \\ W_f \\ W_r
\end{matrix}\right)
\left[\begin{matrix}
   x_1 & x_2 & ... & x_L
\end{matrix}\right]
$$
输出$U$的尺寸是$L\times 3d$，$L$是时域长度，$d$是隐藏节点数。当训练中输入时多batch的数据，那么输出$U$的尺寸是$(L, \quad B,\quad 3d)$

然后就是剩余的逐点运算，针对多batch$B$和隐层节点数维度$d$展开并行。
> Indices：时域长度$L$，batch数$B$，隐层节点数$d$
> Input：batch输入为$x[l,i,j]$，grouped matrix multiplication$U[l,i,j']$，初始状态$c_0[i,j]$，参数$v_f[j],v_r[j],b_f[j],b_r[j]$
> Output：输出$h[.,.,.]$和状态$c[.,.,.]$
> 初始化$h[.,.,.]$和$c[.,.,.]$，尺寸为$L\times B\times d$
> for $i = 1, · · · , B; j = 1, · · · , d$ do    **针对多batch$B$和隐层节点数维度$d$展开并行**
> 	$c = c_0[i, j]$
> 	for$ l = 1, · · · , L$ do
> 		$f = \sigma( U[l, i, j + d] + v_f[j] \circ c + b_f[j] )$
> 		$c = f\circ c + (1 − f)\circ U[l, i, j]$
> 		$r = \sigma( U[l, i, j + 2d] + v_r[j] \circ c + b_r[j] )$
> 		$h = r \circ c + (1 − r) \circ x[l, i, j]$
> 		$c[l, i, j]=c$
> 		$h[l, i, j]=h$
> 返回输出$h[.,.,.]$和状态$c[.,.,.]$



## 3.2 Initialization
训练中增加一个系数避免梯度消失问题
$$
h_t=r_t\circ c_t + (1-r_t)\circ x_t.\alpha
$$
$\alpha$值与bias有关，有公式推导
$$
\alpha=\sqrt{1+2e^b}
$$