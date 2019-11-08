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

全期望公式 $E(E[V\vert W])=E(X)$

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






# Part 2: Generalization Bounds
最小化empirical risk或者训练误差并不能解决学习问题，但是如果$R(h)$和$R_{emp}(h)$的差值足够小，学习问题就被被解决。

这里我们解释为什么这个差值可以做到很小。




## Independently and Identically Distributed 独立同分布
首先做一个假设是，训练数据样本是独立同分布的
    - inference的数据集是从同一个概率分布采样出来，满足同分布
    - 采样的时候不依赖之前或之后的采样结果，样本之间没有依赖关系，满足样本之间相互独立




## 大数定律
**当实验足够多次时，实验输出的平均值，接近真实分布的平均值，这称为大数定律。**

有一随机变量$X$遵循概率分布$P$，从中采样出$m$个独立同分布的样本$x_1,x_2,…,x_m$，那么

$$
\lim_{m\to \infty}⁡P[\vert E_{x\sim P}[X]−\frac{1}{m} \sum_{i=1}^m{x_i} \vert > \epsilon]=0⁡
$$

这个版本称为弱大数定律，它保证当采样数足够大时，样本均值和真实均值**足够接近**，强大数定律版本是，样本均值**就是**真实均值。

应用到泛化误差上，对于某一个hypothesis函数$h$，**只要样本数足够多，基于采样数据集的实验误差$R_{emp}(h)$和基于整个整个概率分布的$R(h)$泛化误差就足够接近**

$$
\lim_{m\to \infty}P[\vert R(h)−R_{emp}(h)\vert > \epsilon]=0
$$




## Hoeffding’s inequality 霍夫丁不等式
大数定律指示指引了原则性方向，但并没有提供任何方法，集中不等式Concentration inequality 提供了方法。

    集中不等式描述了一个随机变量是否集中在某个取值附近:
    - 马尔可夫不等式，从一个随机变量中采样出大于某特定值的概率上限
    - 切比雪夫不等式，从一个随机变量中采样出小于某特定值的概率上限
    - 霍夫丁不等式，从一个随机变量中采样出某个特定区间值的概率上限

有一随机变量$X$遵循概率分布$P$，从中采样出$m$个独立同分布的样本$x_1,x_2,…,x_m$，对于每个$a \leq x_i \leq b$，有

$$
P[\vert E_{x\sim P}[X]−\frac{1}{m} \sum_{i=1}^m{x_i} \vert > \epsilon] \leq 2\exp⁡(\frac{−2m\epsilon^2}{(b−a)^2}
$$

应用到泛化误差上，**empirical risk**定义是

$$
	R_{emp}(h)=\frac{1}{m} \sum_{i=1}^m{L(y_i,hx_i)}
$$

式子$P[\vert E_{x\sim P}[X]−\frac{1}{m} \sum_{i=1}^m{x_i} \vert > \epsilon]$里的$x_i$实际上是测试集误差$L(y_i,hx_i)$，假设测试集误差$L(y_i,hx_i)$被限定在0和1之间，那么$b−a=1$

$$
P[\vert R(h)−R_{emp}(h)\vert > \epsilon] \leq 2\exp⁡(−2m\epsilon^2）
$$

其中$m$是样本数，意味着，随着样本数$m$增大，泛化误差指数下降。

这里的公式都是基于给定hypothesis函数 $h$前提下得到，而学习过程中并不知道hypothesis函数，需要从整个hypothesis空间里找一个。

因此我们需要一个泛化边界来反映挑选正确hypothesis函数的过程。





## Generalization Bound: 1st Attempt
保证整个hypothesis空间都满足泛化误差不超过$\epsilon$，表达式可以重写成

$$
P[sup_{h\in H}⁡ \vert R(h)−R_{emp}(h) \vert > \epsilon]=P[\bigcup_{h\in H}⁡ \vert R(h)−R_{emp}(h)\vert > \epsilon]
$$

$$\bigcup$$表示或运算，也就是，对于hypothesis集$$H$$里的每一个hypothesis函数$$h$$，都满足泛化误差足够小概率

应用布尔不等式得到

$$
P[\bigcup_{h\in H}⁡ \vert R(h)−R_{emp}(h)\vert > \epsilon] \leq \sum_{h\in H} P[\vert R(h)−R_{emp}(h)\vert > \epsilon]
$$

    布尔不等式数学归纳法

$$
\because P(A\bigcap B)\geq0 \\
\therefore P(A\bigcup B)=P(A)+P(B)-P(A\bigcap B)\leq P(A)+P(B)
$$

应用霍夫丁不等式得到

$$
\sum_{h\in H} P[\vert R(h)−R_{emp}(h)\vert > \epsilon] \leq \sum_{h\in H}2\exp⁡(−2m\epsilon^2)=2\vert H \vert exp⁡(−2m\epsilon^2) \\
$$

$\vert H \vert$是hypothesis空间的尺寸，hypothesis函数数目

最终得到，整个hypothesis空间都满足泛化误差不超过$\epsilon$的概率上限

$$
P[sup_{h\in H}⁡ \vert R(h)−R_{emp}(h) \vert > \epsilon] \leq 2\vert H \vert exp⁡(−2m\epsilon^2) \\
$$

令 

$$
2\vert H\vert \exp⁡(−2m\epsilon^2 )=\sigma \\
\epsilon=(\frac{\ln \vert H\vert + \ln \frac{2}{\sigma}}{2m})^\frac{1}{2}
$$

那么给定泛化误差小值$\epsilon$，得到$\sigma=2\vert H\vert \exp⁡(−2m\epsilon^2 )$，

$$
R(h)−R_{emp}(h)\leq \epsilon \\
R(h)−R_{emp}(h)\leq (\frac{\ln \vert H\vert + \ln \frac{2}{\sigma}}{2m})^\frac{1}{2} \\
R(h)\leq R_{emp}(h) + (\frac{\ln \vert H\vert + \ln \frac{2}{\sigma}}{2m})^\frac{1}{2}
$$












