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

Anatomy of High-Performance Matrix Multiplication
>http://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf

# 1. INTRODUCTION
实现接近最优性能的矩阵乘法，不仅需要在宏观层级了解如何将运算拆分成kernel，还需要在微观层级工程实现高性能kernel。这篇paper主要针对宏观问题，也就是如何发现高性能kernel，而不是微观层级的如何设计和实现高性能kernel。
在一个复杂的多级内存架构下，[^1]方法优化减少在相邻层内存之间的数据搬移，与[^2][^3]不同的是，[^1]提出“inner-kernel”概念，对于某些$m_c\times k_c$的矩阵$\tilde{A}$计算$C:=\tilde{A}B+C$，矩阵$\tilde{A}$以某种打包的格式连续存储，并且可以容纳在cache内。但是这里用到的内存结构是不现实的：
- 假设矩阵$\tilde{A}$的inner-kernel计算都在L1 cache
- 忽视了Translation Look-aside Buffer (TLB)问题
最近的[^4]]观察到
- 浮点运算单元能够执行的浮点运算操作数，可以从寄存器流出到L2 cache的浮点数很少。
- 矩阵$\tilde{A}$尺寸的限制因素是TLB能访问的数据量

我们同时发现
- 我们考虑高性能矩阵乘法时需要考虑6种inner-kernel，[^1][^5]只考虑了其中3种

第二章介绍了本文用到的符号，第三章介绍了一种分层的矩阵乘法实现方法，第四章介绍inner-kernel实现。第五章介绍了现实中最常见场景下的算法。第六章给出了现实中如何调整算法参数以优化性能。第七章介绍了不同架构下的算法性能。最后一章包含一些总结。

# 2. NOTATION
矩阵分解矩阵乘法的是核心，给定一个$m\times n$的矩阵$X$，本文中**只考虑按行分块或者按列分块**
$$
X=
\begin{pmatrix}
      X_0 & X_1 & \cdots & X_{N-1}
\end{pmatrix}
=
\begin{pmatrix}
      \check{X}_0 \\
      \check{X}_1 \\
      \cdots \\ 
      \check{X}_{M-1} \\
\end{pmatrix}
$$
这里除了$X_{N-1}$和$\check{X}_{M-1}$以外，$X_j$都有$n_b$列，$\check{X}_i$都有$m_b$行，$X_{N-1}$和$\check{X}_{M-1}$的行列可能会更少。
原始矩阵的乘法将被分解成若干子矩阵的乘法，将三种常见的特殊尺寸的矩阵抽象出来

![Fig3](/assets/Anatomy-of-High-Performance-Matrix-Multiplication/Fig3.png)

![Fig2](/assets/Anatomy-of-High-Performance-Matrix-Multiplication/Fig2.png)

# 3. A LAYERED APPROACH TO GEMM
如下图所示，GEMM可以被分解成多个GEPP，GEMP或者GEPM，进一步可以分解成多个GEBP，GEPB或者GEDOT kernel。
![Fig5](/assets/Anatomy-of-High-Performance-Matrix-Multiplication/Fig5.png)

# 4. HIGH-PERFORMANCE GEBP, GEPB, AND GEPDOT
现在我们讨论GEBP，GEPB和GEDOT的高性能实现。首先我们在一个简单内存结构模型下分析数据搬移开销。更复杂和实际的内存结构模型将在4.2章节讨论。

## 4.1 Basics
下图左边是一个非常简单的多层内存模型，只有寄存器/ cache/RAM。
![Fig6](/assets/Anatomy-of-High-Performance-Matrix-Multiplication/Fig6.png)

在这种简单模型结构下考虑优化GEBP，$C_{m_c,n}+=A_{m_c,k_c}B_{k_c,n}$，其中
- $A$是block，$A\in \Bbb{R}^{m_c\times k_c}$
- $B$是panel，$B\in \Bbb{R}^{k_c\times n}$
- $C$也是panel，$C\in \Bbb{R}^{m_c\times n}$

![Fig7](/assets/Anatomy-of-High-Performance-Matrix-Multiplication/Fig7.png)

3个假设
- Assumptions (a)：$m_c$和$k_c$都足够小，使得cache能够容纳矩阵$A$，矩阵$B$的$n_r$列$B_j$和矩阵$C$的$n_r$列$C_j$
- Assumptions (b)：如果$A$，$B_j$和$C_j$都在cache里，那么$C_j:=AB_j+C_j$能全速利用CPU
- Assumptions (c)：$A$ 可以一直保留在cache里不被切换出去 

基于以上三点假设，上图中GEBP的RAM和cache之间的数据搬移开销为$m_ck_c+k_cn+2m_cn$ memops
- 把$A$从RAM加载到cache，$m_ck_c$
- 把$B_j$从RAM加载到cache，$k_cn$ 
- 把$C_j$从RAM加载到cache，$m_cn$
- 计算完，把$C_j$结果从cache加载到RAM，$m_cn$

而$C_j:=AB_j+C_j$的计算量为$2m_ck_cn$ flops，那么计算量和数据搬移的比例是
$$
\frac{2m_ck_cn}{m_ck_c+k_cn+2m_cn} \approx \frac{2m_ck_cn}{k_cn+2m_cn}=\frac{2m_ck_c}{k_c+2m_c}\frac{flops}{memops} \quad \text{where}\quad m_c<<n
$$

问题变成，**如何选择矩阵$A$的尺寸$m_c$和$k_c$，使得矩阵尺寸满足Assumptions (a)-(c)，而且$\frac{2m_ck_c}{k_c+2m_c}$值最大。**
- 最大化$k_cm_c$，尽可能选择最大尺寸的矩阵$A$，使得$A, B_j, C_j$能够放进cache  
- 在$m_ck_c \leq K$前提下，求$\frac{2m_ck_c}{k_c+m_c}$最大值问题，数学上转换成几何问题，一个长方形边长分别是$2m_c$和$k_c$，在长方形面积受限$m_ck_c \leq K$条件下，如何最大化长方形周长$2(k_c+2m_c)$，这个问题在数学上的最优解就是$k_c=2m_c$，**不应该是方阵?**

现实中$k_c$的选择还受一些其他因素制约，我们将在6.3章节看到。
类似地可以分析GEPB和GEDOT操作。

## 4.2 Refinements
考虑更加实际的应用场景，假设矩阵是column-major order按列存储的。
### 4.2.1 Choosing the cache layer
Fig.6的右边描述了更准确的内存结构，实际上cache也是分成多层的。
首先问题是，尺寸为$k_c\times m_c$的矩阵$A$应该保留在哪一层cache？答案肯定是，在满足Assumptions (a)-(c)的前提下，越快越好，最接近寄存器最快的那一层cache，也就是L1 cache。但是L1 cache天然地非常小，那么可以把矩阵$A$应该保留在L2 cache里，同时让尺寸$k_c\times m_c$更大一些?
$R_{comp}$表示CPU能处理的浮点运算操作速度，$R_{load}$表示CPU能从L2 cache加载到寄存器的速度，假设

- 矩阵$A$保留在L2 cache，而矩阵$B_j$和$C_j$保留在L1 cache
- L1 cache和寄存器之间带宽足够，也就是忽略矩阵$B_j$和$C_j$的加载时间

那么$C_j:=AB_j+C_j$的计算量为$2m_ck_cn_r$ flops，时间开销为$\frac{2m_ck_cn_r}{R_{comp}}$，带宽开销为从L2 cache加载矩阵$A$到寄存器的数据量$m_ck_c$，时间开销为$\frac{m_ck_c}{R_{loa
d}}$。为了克服矩阵$A$的加载时间，必须保证
$$
\frac{2m_ck_cn_r}{R_{comp}} \geq \frac{m_ck_c}{R_{load}} \\
n_r \geq \frac{R_{comp}}{2R_{load}}
$$

**$B_j$和$C_j$保留在L1 cache，$A$保留在L2 cache，希望矩阵计算时间大于L2 cache的加载时间，panel矩阵$B, C$按$n_r$行或$n_r$列拆分成$B_j, C_j$矩阵时，拆分尺寸$n_r$必须足够大，$n_r$的最小值与CPU浮点运算操作速度$R_{comp}$和L2 cache加载到寄存器的速度$R_{load}$相关。**

### 4.2.2 TLB considerations
第二个考量与系统页管理有关。通常我们的系统使用虚拟内存，因此可用内存大小不受实际物理内存大小限制，内存是分页的。有一张page table来映射虚拟地址和物理地址，保持跟踪这一页实在内存还是硬盘。问题是这个表本身可能很大，比如若干Mbytes，妨碍虚拟内存到物理内存的快速转换。为了克服这个问题，一张更小的表Translation Look-aside Buffer (TLB)，用于存储最近用到的表信息。当一个虚拟地址能在TLB里找到时，转换速度是很快的。但没有找到时，称为TLB miss，需要重新访问page table。将信息从page table里拷贝到TLB里。**TLB可以看做是page table的cache。**最近更有些架构中出现类似L2 cache的L2 TLB。
TLB的存在意味着我们需要满足额外的假设

- Assumptions (d)：为了保证计算$C_j:=AB_j+C_j$时不会出现TLB miss，$m_c, k_c$必须足够小使得，block矩阵$A$，panel矩阵$B, C$按$n_r$行或$n_r$列拆分成$B_j, C_j$矩阵可以**同时被TLB访问**
- Assumptions (e)：在直到计算完成之前，block矩阵$A$必须一直能被TLB访问。














# References
[^1]: Gunnels, J. A., Gustavson, F. G., Henry, G. M., and van de Geijn, R. A. 2001. FLAME: Formal linear algebra methods environment. ACM Transactions on Mathematical Software 27, 4 (December), 422–455.
[^2]: Agarwal, R., Gustavson, F., and Zubair, M. 1994. Exploiting functional parallelism of POWER2 to design high-performance numerical algorithms. IBM Journal of Research and Development 38, 5 (Sept.).
[^3]: Whaley, R. C., Petitet, A., and Dongarra, J. J. 2001. Automated empirical optimization of
software and the ATLAS project. Parallel Computing 27, 1–2, 3–35.
[^4]: Goto, K. and van de Geijn, R. A. 2002. On reducing TLB misses in matrix multiplication. Tech. Rep. CS-TR-02-55, Department of Computer Sciences, The University of Texas at Austin.
[^5]: Gunnels, J. A., Gustavson, F. G., Henry, G. M., and van de Geijn, R. A. 2005. A novel
model produces matrix multiplication algorithms that predict current practice. In Proceedings
of PARA’04. Elsevier.





