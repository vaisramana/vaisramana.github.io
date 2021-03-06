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

# inference优化
## 硬件资源评估
首先我们需要指标来评估硬件平台性能和模型或者算法开销，当硬件平台性能满足模型或者算法开销时，我们认为这个模型在这个平台上是可执行的。
评估硬件平台性能或者模型开销一般会用到两个硬件资源指标，算力和带宽

- 算力，单位是FLOP/S，floating-point operations per second，指示这个硬件平台在理想状况下，每秒钟可以完成的浮点运算数。
- 带宽，单位是Byte/s，指示这个硬件平台在理想状况下，每秒可以完成的内存读写量。

那CONV2D为例，输入$$[N,H,W,C_{in}]$$，卷积核$$[K_H,K_W,C_{in},C_{out}]$$，假设padding方式为same，输出$$[N,H,W,C_{out}]$$
- 算力开销为
  $$N\cdot H\cdot W\cdot K_H\cdot K_W\cdot C_{in}\cdot C_{out} \quad {\rm {FLOP}}​$$

- 卷积计算方式的带宽开销为，读输入+读参数+写输出
   $$N\cdot H\cdot W\cdot C_{in} + K_H\cdot K_W\cdot C_{in}\cdot C_{out} + N\cdot H\cdot W\cdot C_{out} \quad {\rm {4 bytes}}​$$

这里假设硬件缓存足够大，所有数据都可以被cache住，只需要读取一次，不需要重复从内存读取。而事实上在嵌入式设备中，这是不可能做到的。

## 计算密度
- 计算密度
  单位是FLOP/byte，加载一个byte可以做多少次运算
下面这个图反应的是，横坐标是计算密度，纵坐标是算力，在一个硬件平台中，一开始算力随着计算密度的增加线性增加，也就是说一开始喂得数据越多，运算处理的越快，但增大到一定程度时，算力就不再增长了。观察到的现象是，一开始随着计算量的增加，运行时间几乎不变，但当算力维持不变时，运行时间就随着计算量的增加线性增加。
前半段曲线是带宽受限，这个区间内我们的目标是提高算力，让模型运行在尽可能接近顶部位置，后半段是算力受限，这个区间内算力已经达到顶点，我们的目标是减低计算量。
  ![Example_of_a_naive_Roofline_model.svg](/assets/accelerating-inference/Example_of_a_naive_Roofline_model.svg.png)

  >https://en.wikipedia.org/wiki/Roofline_model

- 矩阵乘矩阵， $$C_{1000,1000}=A_{1000,1000} \times B_{1000,1000}$$
  - 算力开销，输出每一个点需要1K次乘加操作，一共需要2G FLOP
  - 带宽开销，读两个矩阵，写一个矩阵，每个矩阵4M bytes，至少需要12M bytes
  - 计算密度，2G FLOP/12M bytes = 170 FLOP/MBytes
- 矩阵乘向量，$$C_{1000,1}=A_{1000,1000} \times B_{1000,1}$$
  - 算力开销，输出每一个点需要1K次乘加操作，一共需要2M FLOP
  - 带宽开销，读一个矩阵，忽略读写两个向量，每个矩阵4M bytes，至少需要4M bytes
  - 计算密度，2M FLOP/4M bytes = 0.5 FLOP/MBytes

那么明显矩阵乘矩阵是受限于算力的，矩阵乘向量是受限于带宽的，两者的优化方向是不同的。

## 存储器结构
![ComputerMemoryHierarchy](/assets/accelerating-inference/ComputerMemoryHierarchy.svg)
>https://en.wikipedia.org/wiki/Memory_hierarchy

这是一个计算机的存储器层次结构。越接近CPU的存储器越快也越昂贵。
最接近CPU的寄存器访问速度最快，可以被算数指令直接访问
次一级的cache访问速度较快，容量比寄存器大一些
我们通常意义的数据或者模型都是在RAM里，通常速度是跟不上CPU的速度的。

在带宽受限的场景里，需要合理的运用有限的寄存器和cache资源，使得算力尽可能接近顶点。

举个例子，cache资源在各种计算平台差距很大，NVIDIA tesla V100有36MB片上cache，A113有4KB，MCU平台没有片上cache。
因此普遍的嵌入式设备是没法一次性把3个 $$C_{1000,1000}$$矩阵cache起来，而是需要反复加载同一段内存到cache，用于CPU计算，CPU在很多时候都出于等待状态。因此像矩阵乘矩阵这样的操作，在嵌入式设备上不仅仅受限于算力的，也同样受限于带宽。


## 优化方向


优化方向
 - 降低计算量：fft/winograd加速卷积，稀疏化
 - 降低带宽：gemm，im2col加速卷积
 - 同时降低计算量和带宽：低精度运算

需要注意的是，很多方法是内存占用和计算量/带宽之间的平衡，用更多的内存占用来换取更少的计算量或更低的带宽占用，例如fft/winograd方法，会一定程度上展开内存。在很多嵌入式设备上，没有足够内存来实现这些加速方法，这是加速之外的另一个维度的考量。
NN前向计算中，绝大多数计算量都在于矩阵乘加，比如全连接和RNN操作，完全就是矩阵乘加操作。卷积操作复杂一些，一方面是通过重排，将卷积转换成矩阵乘加，另一方面通过变换，降低计算量。

## GEBP: General block to panel Multiplication
大矩阵分块
- matrix: 两个维度尺寸都很大
- panel: 一个维度尺寸很大
- block: 两个维度尺寸都很小

$$C_{m,n} += A_{m,k} \times B_{k,n}$$

考虑三级结构的简单模型，寄存器/ cache/RAM，假设
- cache足够大，同时放下$$A_{m,k}$$，$$B_{k,n}$$ 的$$n_{r}$$列$$B_{j}$$和$$C_{m,n}$$的$$n_{r}$$列$$C_{j}$$
- $$C_{j}+=AB_{j}$$ 能全速利用CPU
- $$A_{m,k}$$ 一直保留在cache里不会被切换出去
![gebp](/assets/accelerating-inference/gebp.png)

那么
	- 把$$A_{m,k}$$整个放进cache的开销是$$mk$$，
	- 把$$B_{k,n}$$切分成$$n_{r}$$列一份，每次load进去一份开销是$$kn_r$$，一共$$kn$$
	- 把$$C_{m,n}$$切分成$$n_{r}$$列一份，每次load进去一份开销是$$mn_r$$，一共$$mn$$
	- 计算完，把$$C_{m,n_r}$$结果store回的开销是$$mn_r$$，一共$$mn$$
	- 总的内存开销是$$mk+kn+2mn$$
	- 总的计算开销是$$2kmn$$

计算密度近似为，
		$$\frac{2kmn}{mk+kn+2mn}\approx	\frac{2kmn}{kn+2mn}=	\frac{2km}{k+2m} \quad where\quad m<<n$$

让计算密度最大的方法是
- 最大化$$km$$，尽可能选择最大尺寸的$$A_{m,k}$$，使得$$A_{m,k}$$能够放进cache
- 尽量使得$$k==m$$，也就是$$A_{m,k}​$$接近方阵


## GEMM: General Matrix to Matrix Multiplication
回到矩阵乘加，核心思想就是分割，将大矩阵分割成小块或者长条来适配cache，让一次数据读取做尽可能多的运算，提高计算密度。
![gemm](/assets/accelerating-inference/gemm.png)
- 选择cache
当考虑更复杂一些的模型，带多级缓存。我们需要考虑把哪些数据缓存在L1 cache，哪些数据缓存在L2 cache。
比如在VAR2中，分割直到的单元大小适配cache，然后再应用上面提到的GEBP/GEPB等方法
- packing
当一个大矩阵乘法GEMM被拆分成GEBP时，会发生列数据不连续，直接方法会导致大量TLB miss，寻址不连续，这就需要额外的打包工作，将一个小矩阵打包到连续内存。

## im2col
im2col有大神贾扬清实现在caffe，核心思想就是将卷积转换成GEMM
	-  im2col转换，图像转换成矩阵
	-  GEMM计算

![](/assets/accelerating-inference/im2col.png)

- 输入输出的$$HW$$不变，每一个输出的点都需要$$C$$个通道，每个通道$$K\times K$$的输入数据块。如之前所说，为了寻址的连续性，im2col方法预先将$$C \times H\times W$$的数据块连续排列。
- 卷积核也做了转置处理
- 这样卷积操作就转换成矩阵乘法操作，可以应用成熟的GEMM实现。
im2col的缺点就是内存开销增大，对于输入数据进行转换之后扩大了$$k^2$$倍


## winograd/FFT
我们知道时域卷积可以转换成频域点乘，这两种方法本质上是，通过某种线性变换，把卷积核和输入变换到另一个域，FFT是频域，winograd是winograd域，原来空间域下的卷积在新的空间域下变成逐点相乘，再将点乘结果逆变换回原空间域。
	$$Y=A^T[(B^TdB)\bigodot (G^TgG)]A​$$

缺点

- FFT不适合小卷积核，另外非常依赖FFT性能，设备端FFT通常只支持$$2^n$$的长度，对比卷积核的尺寸有要求
- winograd适合小卷积核，比如$$3\times3$$卷积的的优化应用非常广，但是语音中用到的大量非方的卷积核和stride，不适合用winograd

比如在ncnn/featherCNN/NNpack里广泛只用的$$f(6\times 6, 3\times 3)$$的分块方法，将输入矩阵分成$$8\times 8$$的小块，卷积核大小$$3\times 3$$，输出$$6\times 6$$。
	- 按照原始卷积算法，需要$$6\times 6\times (3\times 3)=324$$次乘加
	- $$B_{8,8}^Td_{8,8}B_{8,8}$$需要$$8\times 8\times 8\times 2=1024$$次乘加
	- 点乘需要$$8\times 8=64$$次乘
	- $$Y_{6,6}=A_{6,8}^T[X_{8,8}]A_{8,6}$$需要$$6\times 8\times 8 + 6\times 8\times 6=672$$次乘加
	- 原始 324 vs winograd 1024+64+672=1760

事实上因为转换矩阵$$A_{6,8}B_{8,8}$$都是固定的，我们可以人为选定一些特殊的转换矩阵，使得
	- 出现尽可能多的0
	- 出现尽可能多的重复元素
$$B_{8,8}^Td_{8,8}B_{8,8}$$需要$$8x8x8=512​$$次乘加



将一个浮点数r，量化成定点数q，需要额外两个参数，缩放系数s和定标值z，s是浮点数，Z是定点数，公式就是定点数q减去定标值z乘以缩放系数。另外一种量化方式是先乘后减，这种量化方式已经被弃用，原因是浮点数0用这种方法量化之后有误差，而前一种方式中，定点数z就严格对于浮点数0。浮点数0会在pad中大量运用，因此现在都用前一种量化方式。

考虑网络中最常见的矩阵乘法操作，三个浮点矩阵乘法，用量化方式表示，量化模型的矩阵乘法可以表示成这样，与原始浮点矩阵乘法相比，不仅仍然有$N^3次$乘法和次$N^3次$加法，还增加了次减$2N^3次$法，输出矩阵的每个点都要做$2N$次减法。通过把乘法展开，可以降低运算复杂度。这里对于输入矩阵的行列分别求和，运算复杂度从N^3降低到N^2，因此此时核心的运算量还是在两个定点矩阵的乘法上。
