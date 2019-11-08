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



***原文***

[Machine Learning Theostat_model_simplified.pngry - Part 1: Introduction](https://mostafa-samir.github.io/ml-theory-pt1/)

[Machine Learning Theory - Part 2: Generalization Bounds](https://mostafa-samir.github.io/ml-theory-pt2/)

[Machine Learning Theory - Part 3: Regularization and the Bias-variance Trade-off](https://mostafa-samir.github.io/ml-theory-pt3/)






# Part 1: Introduction

## 定义学习问题
一个监督学习问题里，有一个观察到的数据集$S={(x_1,y_1), ..., (x_m,y_m)}$，其中$x_i$是特征向量，$y_i$是label，学习目的是如何从给定$x_i$推断出$y_i$。
已知条件

- 样本$(x_i, y_i)$是从一个更大集里随机采样出来的，正式表达为，两个随机变量$X$和$Y$分别遵循概率分布$P_X$和$P_Y$，$x$和$y$是这两个随机变量随机出来的数值。
- 特征和label之间存在关联，正式表达为，$Y$是$X$的条件概率$P(Y\vert X)=P(X)P(Y\vert X)$

可以定义一个统计模型来表达学习问题里条件概率分布关系

![](/assets/machine-learning-theory/stat_model.png)

## 目标函数
我们不想直接和条件概率分布打交道，因此引入目标函数。用目标函数来简化代替条件概率函数。
均值和方差可以用来分解一个随机变量，假设有两个随机变量$V$和$W$，

$$
	V=E[V\vert W]+(V−E[V\vert W])
$$

$E[V|W]$是随机变量$V$和$W$的条件均值，上式说明随机变量$V$可以被分解成两部分
- 由另一个随机变量$W$表达部分
- 不能由另一个随机变量 $W$ 表达的随机部分， $Z=V−E[V\vert W]$ ，由全期望公式可知，这部分均值为 $0$， $E(Z)=E(V−E[V\vert W])=E(V)−E(V)=0$ ，因此这部分只包含方差，就是随机变量 $V$ 中不能被随机变量 $W$ 表达的方差 $\xi$


    [全期望公式](https://zh.wikipedia.org/wiki/%E5%85%A8%E6%9C%9F%E6%9C%9B%E5%85%AC%E5%BC%8F)

$$
E(E[V\vert W])=E(X)
$$


考虑一对随机变量$X$和$Y$随机出来的值$(x_i, y_i )$

$$
y_i=E[Y\vert X=x_i ]+\xi
$$

$\xi$是随机变量$Y$中不能被随机变量$X$表达的方差，称为noise项。

定义函数表示条件期望conditional expectation，将输入空间$X$映射到输出空间$Y$，$X\to Y$，$f(x)$称为目标函数，机器学习任务简化成估计函数$f$

$$
E[Y\vert X=x]=f(x)
$$

那么输入特征和输出label的条件概率表达$P(Y\vert X)$可以重写成

$$
	y=f(x)+\xi
$$

![](/assets/machine-learning-theory/stat_model_simplified.png)

## Hypothesis 假设目标函数
在估计目标函数的过程中，因为不能穷举所有函数，我们试图对于目标函数$f(x$)的形式做一个假设hypothesis。

假设目标函数是一个线性函数，或者球面非线性函数，无论做什么假设，都定义一个可能的目标函数空间称为hypothesis set。

比如，我们假设目标函数的形式是$ax+b$，那么hypothesis set $H$就是

$$
	H={h:x\to y\vert h(x)=ax+b}
$$

函数$h(x)=3.2x−1$就是这个hypothesis set $H$里的一个具体实例

机器学习任务变成，从hypothesis set $H$里挑选一个实例目标函数$h$，使得实例目标函数$h$能最接近目标函数$f$。接下来的问题是，如何评估一个hypothesis目标函数接近目标函数？


## 损失函数
在整个数据集应用hypothesis函数$h$，然后求loss的均值，称为in-sample error或者**the empirical risk**

$$
	R_{emp}(h)=\frac{1}{m} \sum_{i=1}^m{L(y_i,hx_i)}
$$

之所以被称为实验性的，是因为这个值来源于数据集中采样出来的实验数据。


## The Generalization Error 泛化误差
学习的目标是数据集的概率分布，而不是数据集本身。

这就意味着hypothesis应该在没见过的采样数据上有小误差，**在整个概率分布上定义泛化误差**

$$
	R(h)=E_{((x,y)\sim P(X,Y))}[L(y_i,h(x_i))]
$$

那么问题是，我们不知道这个联合分布$P(X,Y)$，也就没法计算这个泛化误差$R(h)$


## Is the Learning Problem Solvable?
$R_{emp}(h)$是基于采样数据集的实验误差，$R(h)$是基于整个联合分布$P(X,Y)$的泛化误差，如何在数学上表示两者接近？

两者差值大于一个非常小值的概率足够小

$$
P[sup_{h\in H}\vert R(h)−R_{emp}(h)\vert >\epsilon]
$$

$R(h)$和$R_{emp}(h)$的差值绝对值的上确界 $sup_{h\in H}$，大于一个非常小值$\epsilon$的概率。如果这个概率足够小，说明$R(h$)和$R_{emp}(h$)的足够接近，那么这个学习问题是可解的。


    上确界
    一个实数集合A，若有一个实数M，使得A中任何数都不超过M，那么就称M是A的一个上界。在所有那些上界中如果有一个最小的上界，就称为A的上确界。



















