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




# Learning Structured Sparsity in Deep Neural Networks
> Accepted by NIPS 2016
> University of Pittsburgh
> https://arxiv.org/abs/1608.03665


# Learning Intrinsic Sparse Structures within Long Short-Term Memory
> Published in ICLR 2018
> Duke University & Business AI and Bing Microsoft
> https://arxiv.org/abs/1709.05027




# TinyLSTMs: Efficient Neural Speech Enhancement for Hearing Aids
> Arm ML Research Lab
> [v1] Wed, 20 May 2020
> https://arxiv.org/pdf/2005.11138.pdf
> https://github.com/BoseCorp/efficient-neural-speech-enhancement

## 3. Optimizing LSTMs for HA Hardware
### 3.1. Structural Pruni
分为结构化pruning和随机pruning，随机pruning除非稀疏率非常高，在真实硬件上很难部署。结构化pruning将权重$\theta$分为若干个group $\Gamma$，每个group内包含一组权重$w_g \in \Gamma$。