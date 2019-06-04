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

# Anatomy of High-Performance Matrix Multiplication
>http://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf

## 1. INTRODUCTION

## 2. NOTATION
矩阵分解是核心，将三种特殊尺寸的矩阵抽象出来

![Fig3](/assets/Anatomy-of-High-Performance-Matrix-Multiplication/Fig3.png)

![Fig2](/assets/Anatomy-of-High-Performance-Matrix-Multiplication/Fig2.png)

## 3. A LAYERED APPROACH TO GEMM
GEMM可以被分解成多个GEBP，GEPB，GEDOT。
![Fig5](/assets/Anatomy-of-High-Performance-Matrix-Multiplication/Fig5.png)

## 4. HIGH-PERFORMANCE GEBP, GEPB, AND GEPDOT

![Fig6](/assets/Anatomy-of-High-Performance-Matrix-Multiplication/Fig6.png)

### 4.1 Basics
![Fig7](/assets/Anatomy-of-High-Performance-Matrix-Multiplication/Fig7.png)

考虑三级结构的简单模型，寄存器/ cache/RAM，考虑优化GEBP
	$$ C_{m,n}+=A_{m,k}B_{k,n}$$
其中$$C$$和$$B$$都是panel矩阵，$$A$$都是block矩阵，实现时会将$$C$$和$$B$$都分解成$$N$$个更小的panel $$C_j$$和$$B_j$$，每一个小panel是$$n_{r}$$列  
	- $$A$$一直在cache里  
	- 遍历$$N$$个更小的panel $$C_j$$和$$B_j$$，依次load进cache，与$$A$$乘加并写出

那么这里就有假设  
	- cache足够大，同时放下$$A$$，$$B$$的$$n_{r}$$列$$B_{j}$$和$$C$$的$$n_{r}$$列$$C_{j}$$  
	- $$C_{j}+=AB_{j}$$ 能全速利用CPU  
	- $$A$$ 一直保留在cache里不会被切换出去  

那么计算密度是
	- 把$$A$$整个放进cache的开销是$$mk$$  
	- 把$$B$$切分成$$n_{r}$$列一份，每次load进去一份开销是$$kn_r$$，一共$$kn$$  
	- 把$$C$$切分成$$n_{r}$$列一份，每次load进去一份开销是$$mn_r$$，一共$$mn$$  
	- 计算完，把$$C_{m,n_r}$$结果store回的开销是$$mn_r$$，一共$$mn$$  
	- 总的内存开销是$$mk+kn+2mn$$  
	- 总的计算开销是$$2kmn$$  

计算密度近似为  
    $$\frac{2kmn}{mk+kn+2mn} \approx \frac{2kmn}{kn+2mn}=\frac{2km}{k+2m} \quad \text{where}\quad m<<n$$

问题变成如何选择$$kmn$$让上面的式子值最大  
	- 最大化$$km$$，尽可能选择最大尺寸的$$A_{m,k}$$，使得$$A_{m,k}$$能够放进cache  
	- 尽量使得$$k==m$$，也就是$$A_{m,k}$$接近方阵  

## 4.2 Refinements
假设矩阵是column-major order按列存储的

