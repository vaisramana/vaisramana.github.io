



# Supervised Speech Separation Based on Deep Learning: An Overview

DeLiang Wang, Jitong Chen
(Submitted on 24 Aug 2017 (v1), last revised 15 Jun 2018 (this version, v2))

## I.INTRODUCTION
语音分离的目的是从背景噪声中分离目标语音。



## II.CLASSIFIERS AND LEARNING MACHINES



## III.TRAINING TARGETS
主要有两类训练目标
- masking-based targets
- mapping-based targets

语音分离衡量标准
- 信号层级的
目标是量化语音增强或者干扰消除的程度，除了传统饿SNR，还有语音失真和噪声残留，还包括 SDR (source-to-distortion ratio), SIR (source-tointerference ratio), and SAR (source-to-artifact ratio)
- 听觉层级的

![](/assets/Supervised-Speech-Separation-Based-on-Deep-Learning-An-Overview/training_targets.png)

#### A. Ideal Binary Mask
基于一段带噪信号的二维$T-F$表达，比如听觉谱 Cochleagram或者语谱图 Spectrogram，做如下二分类

$$
IBM=\begin{cases}
     1,  & \text{if $SNR(t,f)>LC$} \\
     0, & \text{otherwise}
     \end{cases}
$$

其中$LC$是一个阈值，产生label的时候需要对频谱上的每一个$(T,F)$点做标注，check是否是0还是1，这是一个有监督的分类任务，loss函数一般用交叉熵。

#### B. Target Binary Mask
类似IBM也是做二分类，和IBM不同的是，label来源不是SNR，而是每个$(T,F)$点的语音信号能量是否超过固定干扰。

#### C. Ideal Ratio Mask
IBM和TBM属于hard label，IRM属于soft label

$$
IRM=(\frac{S(t,f)^2}{S(t,f)^2+N(t,f)^2})^{\beta}
$$

$S(t,f)^2$和$N(t,f)^2$表示每个$(T,F)$点的语音信号能量和噪声能量，$\beta$是可调参数。
- 当$\beta=0.5$时，相当于做开方，能很好的维持语音信号，这里假设$S(t,f)^2$和$N(t,f)^2$是不相关的，这个假设在非加性噪声场景就不适用，比如混响场景。
- 当$\beta=1$，就类似经典的维纳滤波器

IRM的loss函数通常会用MSE。

#### D. Spectral Magnitude Mask
SMM或者FFT-MASK是基于干净语音和带噪语音的短时傅里叶变换STFT，

$$
SMM(t,f)=\frac{\vert S(t,f) \vert}{\vert Y(t,f) \vert}
$$

$\vert S(t,f) \vert$和$\vert Y(t,f) \vert$表示干净语音和带噪语音的频谱幅值。和IRM不同的是，SMM取值没有限制在1以内。为了获得分离后的语音，我们在频域幅值上应用SMM或者它的估计，然后再合成出分离后的语音。

#### E. Phase-Sensitive Mask
PSM在SMM的基础上做了扩展

$$
PSM(t,f)=\frac{\vert S(t,f) \vert}{\vert Y(t,f) \vert} \cos \theta
$$

$\theta$表示干净语音相位和带噪语音相位的相位差。引入相位差带来更高的SNR，比SMM估计出的干净语音更好。

#### F. Complex Ideal Ratio Mask
cIRM是IRM的复数域版本，与IRM相比能更好地从带噪语音里重建干净语音

$$
S=cIRM * Y
$$

这里$S$和$Y$是干净语音和带噪语音的STFT，$*$表示复数乘法。*cIRM*计算如下

$$
cIRM=\frac{Y_rS_r+Y_iS_i}{Y_r^2+Y_i^2}+i\frac{Y_rS_i-Y_iS_r}{Y_r^2+Y_i^2}
$$

其中$Y_r$和$Y_i$是带噪语音的实部和虚部，$S_r$和$S_i$是干净语音的实部和虚部，因此参数$cIRM$也是一个复数。

#### G. Target Magnitude Spectrum
从带噪语音里直接估计干净语音的频谱，这里频谱可能是幅值谱，也可能是mel谱，通常会去log来压缩动态范围。TMS的形式是取对数而且归一化的频谱。
loss函数是MSE。

#### H. Gammatone Frequency Target Power Spectrum
与TMS不同的是，频谱是基于伽马滤波器的听觉谱。

#### I. Signal Approximation
SA的想法是，训练一个ratio mask来最小化干净语音和估计语音的频谱幅值差值。

$$
SA(t,f)=[RM(t,f)\vert Y(t,f) \vert - \vert S(t,f) \vert]^2
$$

$RM(t,f)$是SMM中的ratio mask，因此SA可以被看做是，ratio mask和spectral mapping的组合，目标是寻求最大的SNR。
训练方式two-stage
- 用SMM做target训练
- fine-turn来减少波形之间的幅值差值 $SA(t,f)$




## IV.FEATURES
- mel-domain features
	- mel-frequency cepstral coefficient (MFCC)
	- delta-spectral cepstral coefficient (DSCC) 
- linear prediction features
	- perceptual linear prediction (PLP) 
	- relative spectral transform PLP (RASTA- PLP)
- gammatone-domain features
	- gammatone feature (GF)
	- gammatone frequency cepstral coefficient (GFCC)
- zero-crossing features
	- zero-crossings with peak-amplitudes (ZCPA)
- autocorrelation features
	- relative autocorrelation sequence MFCC (RAS-MFCC) 
	- autocorrelation sequence MFCC (AC-MFCC)
	- phase autocorrelation MFCC (PAC-MFCC)
- medium-time filtering features
	- power normalized cepstral coefficients (PNCC) 
	- suppression of slowly-varying components and the falling edge of the power envelope (SSF)
- modulation domain features
	- Gabor filterbank (GFB)
	- amplitude modulation spectrogram (AMS)
- Pitch-based (PITCH) features

用group Lasso方法选择特征，推荐特征集合包括AMS, RASTA- PLP和MFCC。
这篇paper提出Multi-Resolution Cochleagram (MRCG) ，基于频谱不同精度计算四个cochleagrams，保证提供足够的本地和全局contex。然后提出的特征用auto-regressive
moving average (ARMA)滤波器做后处理，用HIT−FA rate作为衡量指标
>HIT−FA rate
>HIT指的是语音点被正确分类，FA指的是噪音点呗错误分类

特征选择对于性能影响很大。

![](/assets/Supervised-Speech-Separation-Based-on-Deep-Learning-An-Overview/classification_performance.png)


## V.MONAURAL SEPARATION ALGORITHMS
单通道语音增强，去混响，去噪，说话人分离。

### A. Speech Enhancement

### B. Generalization of Speech Enhancement Algorithms
考虑三个方面的泛化能力

- 噪声
噪音有平稳和非平稳，训练中选择的噪音总是有限的



paper[^23]提出的方法是noise perturbation，或者说frequency perturbation，**频谱上的每个点在纵轴频域方向随机跳变**，方法如下
对于每一个T-F点赋一个随机值，服从均匀分布
$$
r(f,t)\sim U(-1,1)
$$

根据当前点附近的点的随机值，计算出perturbation factor $\delta(f,t)$

$$
\delta(f,t) = \frac{\lambda}{(2p+1)(2q+1)} \sum_{f'=f-p}^{f+p} \sum_{t'=t-q}^{t+q} r(f',q')
$$

这里$p$和$q$控制平滑区域大小，$lambda$控制perturbation之后的幅值。这些都是超参数。最终perturbation之后的频谱可以写成

$$
S'(f,t)=S(f+\delta(f,t), t)
$$

- 说话人

- SNR
可以在训练中多增加几组SNR值以提高SNR泛化能力。事实上泛化能力对SNR并不敏感，虽然训练过程中只有有限几组SNR组合，但是帧级别和频谱点级别的SNR变化很大，提供了足够的多样性来支撑泛化能力。还有一种方法是在训练中逐步增大隐层节点数，来适应更低的SNR条件。



## REFERENCES
[^23]: J. Chen, Y. Wang, and D.L. Wang, "Noise perturbation for supervised speech separation," Speech Comm., vol. 78, pp. 1-10, 2016. 






