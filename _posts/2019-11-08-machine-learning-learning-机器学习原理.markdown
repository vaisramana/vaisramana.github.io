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

[Machine Learning Theory - Part 1: Introduction](https://mostafa-samir.github.io/ml-theory-pt1/)

[Machine Learning Theory - Part 2: Generalization Bounds](https://mostafa-samir.github.io/ml-theory-pt2/)

[Machine Learning Theory - Part 3: Regularization and the Bias-variance Trade-off](https://mostafa-samir.github.io/ml-theory-pt3/)






# Part 1: Introduction

## 定义学习问题
一个监督学习问题里，有一个观察到的数据集$$S={(x_1,y_1), ..., (x_m,y_m)}$$，其中$$x_i$$是特征向量，$$y_i$$是label，学习目的是如何从给定$$x_i$$推断出$$y_i$$。
已知条件

- 样本$$(x_i, y_i)$$是从一个更大集里随机采样出来的，正式表达为，两个随机变量$$X$$和$$Y$$分别遵循概率分布$$P_X$$和$$P_Y$$，$$x$$和$$y$$是这两个随机变量随机出来的数值。
- 特征和label之间存在关联，正式表达为，$$Y$$是$$X$$的条件概率$$P(Y|X)=P(X)P(Y|X)$$

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

>[全期望公式](https://zh.wikipedia.org/wiki/%E5%85%A8%E6%9C%9F%E6%9C%9B%E5%85%AC%E5%BC%8F)
>$$
>E(E[V\vert W])=E(X)
>$$




























