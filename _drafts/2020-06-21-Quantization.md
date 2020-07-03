

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









# Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks
> IEEE ICCV 2019
> https://arxiv.org/abs/1908.05033

Differentiable Soft Quantization (DSQ)用双曲正切函数来代替低比特量化的楼梯函数，保证可微分。

ARM平台部署时，MLA指令会将两个8bit寄存器相乘并累加到另一个bit寄存器中，考虑到保存累加的寄存器可能溢出，会用SADDW指令再转换到16bit寄存器中。
SADDW会有额外的计算开销，需要尽可能减少SADDW的次数，考虑b-bit带符号数的量化乘加操作，两个b-bit带符号数乘法的最大值是$(-2^b-1)^2$，那么8-bit的寄存器只能保证$\frac{2^7-1}{(-2^b-1)^2}$次运算不溢出，可见量化bit数越小，节省的SADDW次数越多

| b         | 2    | 3    | 4    |
| --------- | ---- | ---- | ---- |
| MLA/SADDW | 31/1 | 7/1  | 1/1  |

可见4bit量化每次MLA结果就必须用16bit存储，而2bit量化可以做到累加31次MLA才转换到16bit。
