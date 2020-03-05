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
https://arxiv.org/pdf/1607.01097.pdf
https://github.com/tensorflow/adanet

# 1. Introduction
训练神经网络过程中，我们需要指定网络结构，指定参数，然后解决一个非凸优化问题。
> 凸优化和非凸优化
>  - 凸优化有个非常重要的定理，即任何局部最优解即为全局最优解。由于这个性质，只要设计一个较为简单的局部算法，例如贪婪算法（Greedy Algorithm）或梯度下降法（Gradient Decent），收敛求得的局部最优解即为全局最优。
>  - 非凸优化问题被认为是非常难求解的，因为可行域集合可能存在无数个局部最优点，通常求解全局最优的算法复杂度是指数级的（NP难）

![](/assets/AdaNet/nonconvex-opt.jpg)

从优化角度看，这种方法不能保证模型最佳，通常需要一些特别的手段比如梯度clipping或者BN。
不同的机器学习任务对应不同的模型复杂度，在



机器学习中的重要步骤是选择网络结构，包括层数和每层节点数。复杂度高的网络结构加上不足的数据可能造成过拟合，而负责度低的网络结构可能满足不了性能需求。用户负责来决定一个有着合适复杂度的网络结构，而这通常需要相当高的专业知识。因此，典型的做法是用验证集来选择超参数。用grid search或者random search通常开销非常大。
本文介绍了同时学习网络结构和参数的学习问题的理论分析，据我们所知，这是第一个structural learning of neural networks问题的generalization bounds，这些general guarantees可以指导我们设计不同的算法。
不仅是增强给定网络复杂度下的一个预定义的网络结构，ADANET可以自适应地学习合适的网络结构。 从简单的线性模型开始，算法会逐步添加节点和层来增加网络复杂度。子网络的选择依赖于它们的复杂度和受learning guarantees指导。更重要的是，我们算法得到的优化问题都是凸的，因此保证获得唯一的全局解。

# 2. Network architecture
这一章节我们描述通用网络结构，同时定义我们的hypothesis集。为了简化表达，我们目前只关注二分类问题，后面证明可以推广到多分类问题。
一个普通的前馈神经网络是，多层结构，每层节点只和下一层节点相连。这里我们考虑一种更通用的结构，每一个节点可以和下层的任意节点相连。如下图所示
![](/assets/AdaNet/fig1.jpg)

事实上，这里的定义包含所有的有向不循环图directed acyclic graph (DAG)。
更正式地用公式定义神经网络，假设$l$是中间层层数，$n_k$是第$k$层$k\in[l]$的最大节点数，其中第$k$层的节点$j$，$j\in[n_k]$，可以被一个$h_{k,j}$表示的函数表示，不包含激活函数。由$\cal{X}$表示输入样本空间，对于任意输入$x\in \cal{X}$，让$\Psi(x) \in \Bbb{R}^{n_0}$表示$x$对应的特征向量，$n_0$表示是0层节点数，也就是整个网络的输入。
那么第一层函数$h_{1,j}$，$j\in[n_1]$，的集合$\cal{H_1}$可以表示为
$$
\cal{H_1}=\{x\mapsto u\cdot \Psi(x):u\in \Bbb{R}^{n_0},||u||_{p}\leq \Lambda_{1,0}\}
$$
这里$u$表示一个layer 0和layer 1之间的权重，$p \geq 1$定义了一个$l_p$-norm，$l_p$范数公式
$$
||x||_p=(||x_1||^p+||x_2||^p+ \dots + ||x_n||^p)^{\frac{1}{p}}
$$
当$l_p=2$时就是欧氏距离
$$
||x||_2=(||x_1||^2+||x_2||^2+ \dots + ||x_n||^2)^{\frac{1}{2}}
$$
$||u||_{p}\leq \Lambda_{1,0}$就是保证layer 0和layer 1之间权重$u$的$l_p$范数小于一个超参数$\Lambda_{1,0}$，**$\Lambda_{1,0}$和$p$是调节网络稀疏性和复杂性的超参数**。
$x\mapsto u\cdot \Psi(x)$表示输入$x$经过$\Psi(x)$转换成特征向量，再经过layer 0和layer 1之间权重$u$，得到layer 1的输出。

从layer 1扩展到任意层有
$$
\cal{H_k}=\{x\mapsto \sum_{s=1}^{k-1}u_s\cdot (\varphi_s \circ h_s)(x) :u\in \Bbb{R}^{n_0},||u_s||_{p}\leq \Lambda_{k,s},h_{k,s}\in \cal{H_s}\}
$$
对于每个单元函数$h_{k,s}$，
$\varphi_s \circ h_s=\varphi_s \circ h_{s,1}, \dots, \varphi_s \circ h_{s,n_s}$，其中 $\varphi_s$是一个1-Lipschitz激活函数，比如RELU或者sigmoid。
比如对于layer 1的单元函数$\cal{H_2}$有
$$
\cal{H_k}=\{x\mapsto u_1\cdot (\varphi_1 \circ h_1)(x) :u\in \Bbb{R}^{n_0},||u_1||_{p}\leq \Lambda_{2,1},h_{2,1}\in \cal{H_1}\}
$$
$u_1\cdot (\varphi_1 \circ h_1)(x)$表示输入$x$经过layer 1单元函数$h_1$之后，再经过激活函数$\varphi_1$转换成layer 2的输入，经过layer 1和layer 2之间权重$u_1$，得到layer 2的输出。





前面讨论的网络结构中，输出节点可以连接任意一个中间节点，表示为
$$
f=\sum_{k=1}^{l}\sum_{j=1}^{n_k}w_{k,j}h_{k,j}=\sum_{k=1}^{l} {\bf w}_k {\bf h}_k
$$




















