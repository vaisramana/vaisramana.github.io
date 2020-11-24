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


The Deep Learning Compiler: A Comprehensive Survey
https://arxiv.org/abs/2002.03794

# 1 INTRODUCTION
深度学习硬件可以分为
1. general-purpose hardware with software-hardware co-design
通用硬件比如CPU, GPU加入特殊的硬件模块比如AVX512向量单元
2. dedicated hardware fully customized for DL models
专用硬件比如google TPU
3. neuromorphic hardware inspired by biological brain scienc

通用硬件方面，高度优化的线性代数库Basic Linear Algebra Subprograms (BLAS) libraries比如MKL和cuBLAS，可以作为DL计算基础。另外，硬件厂家发布一些特殊优化的经过裁剪的库用于DL计算，比如MKL-DNN或者cuDNN，可以同时支持前向和后向计算。专用硬件方面，也有些库来加速计算，比如TensorRT支持图优化和低精度量化。
DL库和工具的缺点是，通常开发速度慢于DL模型的发展，因此在新模型上不能充分利用硬件资源。

为了解决DL库和工具的缺点，同时为了缓和DL模型和DL硬件的手动优化压力，出现了DL编译器，比如TVM，Tensor Comprehension, Glow, nGraph and XLA。
DL编译器用DL框架产生的模型作为输入，生成适配不同硬件的代码实现。特定模型和特定代码之间必须是
- 高度目标模型优化
- 高度目标硬件优化

和传统编译器类似，DL编译器也分为
- 前端
- IR
- 后端

本文贡献
- 从各方面比较DL编译器，包括硬件支持，DL架构支持，代码生成和优化
- 分析DL编译器架构，提出核心设计模块，比如多层IR，前端优化比如节点级别，block级别和dataflow级别，和后端优化比如特定硬件优化，auto-tuning和优化kernel。
- 未来DL编译器发展，比如动态shape，前后处理，子图，量化等


# 2 BACKGROUND
## 2.1 Deep Learning Frameworks
## 2.2 Deep Learning Hardware
## 2.3 Hardware-specific DL Code Generator
CPU/GPU可编程性强但能耗高，ASIC能耗低但可编程性弱，FPGA成为中间的桥梁。High-Level Synthesis (HLS)编程模型能让FPGA工程师用高级语言比如C/C++产生高效的硬件设计，避免写verilog或者VHDL，降低编程门槛。Xilinx Vivado HLS and Intel FPGA SDK for OpenCL是两个最流行的HLS工具。DL模型映射到FPGA的两个问题是
- DL模型一般是由DL框架语言描述的，而不是C/C++
- DL相关的信息和优化很难被利用





