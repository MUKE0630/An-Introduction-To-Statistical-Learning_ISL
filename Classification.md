# Classification

# 分类概述

# 为什么不使用线性回归

使用独热编码进行线性回归，编码方式暗含了对结果的排序，不同的编码方式，会产生不同的线性模型，在测试集上出现不同的预测值。这种情况在两种水平的定性问题上会稍好一些。

不适用回归方法进行分类的原因：

- 回归方法不能容纳具有两个以上类别的定性响应
- 回归方法不能提供 $Pr(Y|X)$ 的有意义估计

# 逻辑回归

> 逻辑回归不直接对响应变量 Y 进行建模，而是对 Y 属于某个特定类别的概率建模
> 

### 逻辑回归模型

- 逻辑函数
    
    > 使用函数 $p(X)$ 进行建模，该函数为所有的 $X$ 提供 0~1之间的输出
    > 
    
    $p(X)=\frac{e^{\beta_{0}+\beta_{1} X}}{1+e^{\beta_{0}+\beta_{1} X}}$
    
    使用极大似然法来拟合该模型。
    
    该模型进一步可得
    
    $\frac{p(X)}{1-p(X)}=e^{\beta_{0}+\beta_{1} X}$
    
    量 $\frac{p(X)}{1-p(X)}$ 称为几率，可以取0到∞之间的任意值。两边取对数有
    
    $\log \left(\frac{p(X)}{1-p(X)}\right)=\beta_{0}+\beta_{1} X$
    
    左边称为 logit ，是一个关于 $X$ 的线性量
    
    $X$ 与 $p(X)$ 的关系不是一条直线，$p(X)$ 每变化一个单位，$X$ 的变化率取决于 $X$ 的当前值，从下图中可看出
    
    ![Untitled](Classification%20de1ade1befa04bc79200cd723ad36f2f/Untitled.png)
    

### 回归系数的估计

我们寻求 $\beta_0$ 和 $\beta_1$ 的估计值，使得每个个体的概率尽可能与观测的状态一致。逻辑回归模型中使用的似然函数的数学形式为

$\ell\left(\beta_{0}, \beta_{1}\right)=\prod_{i: y_{i}=1} p\left(x_{i}\right) \prod_{i^{\prime}: y_{i^{\prime}}=0}\left(1-p\left(x_{i^{\prime}}\right)\right)$

选择合适的 $\hat{\beta_0}$  和 $\hat{\beta_1}$ 来最大化这个似然函数

### 做出预测

一旦估计出系数，我们就可以计算出任意给定 $X$ 的概率 $\hat{p}(X)$

### 多元逻辑回归

可以将[该式子](Classification%20de1ade1befa04bc79200cd723ad36f2f.md)推广为

$\log \left(\frac{p(X)}{1-p(X)}\right)=\beta_{0}+\beta_{1} X_{1}+\cdots+\beta_{p} X_{p}$    也可以写为  

$p(X)=\frac{e^{\beta_{0}+\beta_{1} X_{1}+\cdots+\beta_{p} X_{p}}}{1+e^{\beta_{0}+\beta_{1} X_{1}+\cdots+\beta_{p} X_{p}}}$

同样使用最大似然法估计系数

### 多项式逻辑回归

在这之前逻辑回归方法只允许 K=2 类的响应变量（即最终预测结果为 0 或 1）

首先选择单个多项式逻辑回归类作为基线（baseline）为这个角色选择第 K 类
$\operatorname{Pr}(Y=k \mid X=x)=\frac{e^{\beta_{k 0}+\beta_{k 1} x_{1}+\cdots+\beta_{k p} x_{p}}}{1+\sum_{l=1}^{K-1} e^{\beta_{l 0}+\beta_{l 1} x_{1}+\cdots+\beta_{l p} x_{p}}}$    k = 1，2，……，K-1

$\Pr(Y=K|X=x)=\dfrac{1}{1+\sum_{l=1}^{K-1}e^{\beta_{l0}+\beta_{l1}x_1}+\cdots+\beta_{l_p}x_p}$

$\log\left(\dfrac{\Pr(Y=k|X=x)}{\Pr(Y=K|X=x)}\right)=\beta_{k0}+\beta_{k1}x_1+\cdots+\beta_{kp}x_p$

我们引入 softmax 编码，即无论编码如何，任意一对类之间的拟合值（预测值）、对数优势和其他关键模型输出都将保持不变。但是softmax编码在机器学习文献(并将在第10章再次出现)的一些领域被广泛使用，因此值得关注。在softmax编码中，我们不是选择一个基线类，而是对称地对待所有的 K 类，并假设对于k = 1，……，K

$Pr(Y=k|X=x)=\frac{e^{\beta_{k0}+\beta_{k1}x_1+\cdots+\beta_{kp}x_p}}{\sum_{l=1}^Ke^{\beta_{l0}+\beta_{l1}x1+\cdots+\beta_{lp}x_p}}$

由此公式推出，第 $k$ 类与第 $k'$ 类之间的对数优势比（log odds ratio）相等

$\log\left(\dfrac{\Pr(Y=k|X=x)}{\Pr(Y=k'|X=x)}\right)=(\beta_{k0}-\beta_{k'0})+(\beta_{k'1}-\beta_{k'1})x_1+\cdots+(\beta_{kp}-\beta_{k'p})x_p$

# 生成式模型分类

在已有逻辑回归的情况下，为什么还要别的方法？

- 当两类之间存在实质性分离时，Logistic回归模型的参数估计出现异常不稳定。
- 如果预测变量X在每个类别中的分布近似正态分布，且样本量较小，那么本节的方法可能比逻辑回归更准确
- 本节中的方法可以自然地推广到两个以上响应类的情况。

假设分类的观测值为 $K$（$K\ge2)$类，即 $Y$ 有 K 个不同且无序的值；令 $\pi_k$ 表示随机选择的观测来自前 k 类的总体或先验概率；令 $f_k(X)\equiv Pr(X|Y=k)$ 表示来自第 k 类观测的 X 的密度函数；也就是说，如果第 k 类中的观测有很大概率 $X≈x$，则 $f_k ( x )$ 比较大；如果第 k 类中的观测有 $X≈x$ 的可能性很小，则 $f_k ( x )$ 比较小。则**贝叶斯定理**指出：

$Pr(Y=k|X=x)=\frac{\pi_kf_k(x)}{\sum_{l=1}^K\pi_lf_l(x)}$  …………  1

将使用缩写 $p_k(x)=Pr(Y=k|X=x)$ 这就是观测后验 $X=x$ 属于第 k 类的后验概率，即给定该观测的预测值，它是该观测属于第 k 类的概率

一般来说，如果我们有来自总体的随机样本，估计k是很容易的：我们简单地计算属于第k类的训练观测值的分数。然而，估计密度函数 $f_k ( x )$ 要困难得多。接下来有几种方法来估计 $f_k(x)$ ，然后带入上述方程逼近 Bayes 类

### p = 1 时的线性判别法

> 假设 p = 1，即只有一个预测因子。为了估计 $f_k(x)$ ，首先对形式做出一些假设：**假设 $f_k(x)$ 是正态分布或高斯分布，**在一维的设定中，正态密度的形式为
> 

$f_k(x)=\dfrac{1}{\sqrt{2\pi}\sigma_k}\exp\left(-\dfrac{1}{2\sigma_k^2}(x-\mu_k)^2\right)$  …………  2

$\mu_k$ 和 $\sigma_k^2$ 分别为第 k 类的均值和方差。进一步假设 $\sigma_1^2=\cdots=\sigma_K^2$ ，即所有的 K 类都有一个共享的方差项，用 $\sigma^2$ 来表示。公式 2 带入 1 中：

$p_k(x)=\dfrac{\pi_k\frac{1}{\sqrt{2\pi\sigma}}\:\exp\left(-\frac{1}{2\sigma^2}(x-\mu_k)^2\right)}{\sum_{l=1}^K\pi_l\frac{1}{\sqrt{2\pi}\sigma}\:\exp\left(-\frac{1}{2\sigma^2}(x-\mu_l)^2\right)}$ ………… 3

$\pi_k$ 表示观测值属于第 k 类的先验概率。对公式 3 取对数并重新排列项得公式 4，不难看出，这相当于将观测分配给了其中的最大类

$\delta_k(x)=x\cdot\dfrac{\mu_k}{\sigma^2}-\dfrac{\mu_k^2}{2\sigma^2}+\log(\pi_k)$ ………… 4

例如，若 $K = 2$ 且 $\pi_1$ = $\pi_2$，则当 $2x( \mu_1-\mu_2 ) > \mu_1^2 - \mu_2^2$ 时，Bayes类将一个观测分配给类1，否则分配给类2，Bayes决策边界为 $\delta_1 ( x ) = \delta_2 ( x )$ 的点，可以证明这等于

$x=\dfrac{\mu_1^2-\mu_2^2}{2(\mu_1-\mu_2)}=\dfrac{\mu_1+\mu_2}{2}$   ……………  5

我们知道 $X$ 取自每个类内的高斯分布，并且我们知道所涉及的所有参数，在这种情况下，我们可以计算贝叶斯分类，。在实际生活中，我们无法计算贝叶斯分类

在实际应用中，即使我们很确定我们的假设是X取自每个类内的高斯分布，为了应用贝叶斯分类，我们仍然需要估计参数 $\mu_1,\dots,\mu_K$，$\pi_1,\dots,\pi_K$ 和 $\sigma^2$ 。线性判别分析( LDA )方法通过将 $\pi_k$，$\mu_k$ ，$\sigma^2$ 的估计值代入式 4 来近似线性判别分析Bayes类。特别地，使用了以下估计：

$\hat{\mu}_k\quad=\quad\dfrac{1}{n_k}\sum_{i:y_i=k}x_i$

$\hat{\sigma}^2\quad=\quad\dfrac{1}{n-K}\sum_{k=1}^K\sum_{i:y_i=k}(x_i-\hat{\mu}_k)^2$  ………… 6

其中 $n$ 为训练观测总数，$n_k$ 为第 $k$ 类的训练观测数。对 $\mu_k$ 的估计仅仅是来自第 $k$ 类的所有训练观测值的平均值，而 $\hat{\sigma}^2$ 可以看作是对每个$K$ 类的样本方差的加权平均。有时我们知道类成员概率 $\pi_1,\dots,\pi_K$，可直接使用。在没有任何额外信息的情况下，LDA使用属于第 $k$ 类的训练观测值的比例来估计 $\pi_k$，即

$\hat{\pi}_k=n_k/n$ …………  7

LDA类将公式 6 和公式 7 中给出的估计代入公式 4 得到公式 8，并为一个观测 $X = x$ 分配到最大的类

$\hat{\delta}_k(x)=x\cdot\dfrac{\hat{\mu}_k}{\hat{\sigma}^2}-\dfrac{\hat{\mu}_k^2}{2\hat{\sigma}^2}+\log(\hat{\pi}_k)$  …………  8

需要重申的是，LDA分类的结果是假设每个类内的观测值来自具有类均值和**共同方差** $\sigma^2$ 的正态分布，并将这些参数的估计值插入贝叶斯分类

### p>1时的线性判别法

> 现在将LDA分类器扩展到多个预测变量的情况。为了做到这一点，我们假设$X = ( X_1, X_2,\cdots ,Xp)$是从多元高斯(或多元正态)分布中得出的，**具有特定类别的多元高斯均值向量和共同的协方差矩阵**
> 

多元高斯分布假设每个单独的预测变量服从一维正态分布，每一对预测变量之间具有一定的相关性。

为了说明 $p$ 维随机变量 $X$ 具有多元高斯分布，我们记 $X\sim N(\mu,\boldsymbol{\Sigma})$ 其中，$E ( X ) =\mu$ 为 $X$ (具有p个分量的向量)的均值，$Cov ( X ) =\boldsymbol{\Sigma}$ 为 $X$

的 $p × p$ 协方差矩阵。形式上，多元高斯密度定义为：

${f(x)=\dfrac{1}{(2\pi)^{p/2}|\mathbf{\Sigma}|^{1/2}}\exp\left(-\dfrac{1}{2}(x-\mu)^{T}\mathbf{\Sigma}^{-1}(x-\mu)\right)}$ 
  …………  1

在p > 1个预测变量的情况下，LDA分类器假设第 $k$ 类中的观测值取自多元高斯分布 $N(\mu,\boldsymbol{\Sigma})$，其中 $k$ 是特别类别的均值向量，是所有 $K$ 类共有的协方差矩阵。将第 $k$ 类的密度函数 $f_k( X = x)$ 代入式 [( 4.15 )](Classification%20de1ade1befa04bc79200cd723ad36f2f.md) 并执行一点代数运算，可以发现Bayes类将一个观测值 $X = x$ 分配给它所在的类:

$\delta_k(x)=x^T\boldsymbol{\Sigma}^{-1}\mu_k-\dfrac{1}{2}\mu_k^T\boldsymbol{\Sigma}^{-1}\mu_k+\log\pi_k$

这是式[( 4.18 )](Classification%20de1ade1befa04bc79200cd723ad36f2f.md)的向量/矩阵版本

LDA正试图逼近所有分类器中总错误率最低的Bayes分类器。也就是说，不管错误来自哪个类别，贝叶斯分类都会产生最小的错误分类观测值总数。可以对LDA进行修改，以便开发出更符合需求的分类器。

贝叶斯分类器通过给后验概率 $p_k ( X )$ 最大的类分配一个观测来工作。在二分类情况下，这相当于给默认类分配一个观测，如果

$\Pr(\operatorname{default}=\operatorname{Yes}|X=x)>0.5$

然而，如果我们担心对违约个体错误地预测违约状态，那么我们可以考虑降低阈值

$\Pr(\operatorname{default}=\operatorname{Yes}|X=x)>0.2$

如何确定哪个阈值是最好的?这样的决策必须基于领域知识，例如与违约相关的成本的详细信息。

ROC曲线是同时显示所有可能阈值的两类误差的流行图

![Untitled](Classification%20de1ade1befa04bc79200cd723ad36f2f/Untitled%201.png)

一个分类器的总体性能，总结了所有可能的阈值，由( ROC )曲线下面积( AUC )给出，理想的ROC曲线会拥抱左上角，AUC越大，分类器越好。

### 二次判别分析(Quadratic Discriminant Analysis , QDA)

> 与LDA一样，QDA分类的结果是假设每个类别的观测值来自高斯分布，并将参数的估计值插入贝叶斯定理以进行预测。然而，与LDA不同，QDA假设每个类都有自己的协方差矩阵。也就是说，它假设来自第 $k$ 类的观测是$X\sim N(\mu_k,\boldsymbol{\Sigma}_k)$，其中 $k$ 是第 $k$ 类的协方差矩阵。
> 

在此假设下，贝叶斯分类器为最大的类分配一个观测 $X = x$

$\begin{array}{r c l}{{\delta_{k}(x)}}&{{=}}&{{-\frac{1}{2}(x-\mu_{k})^{T}\Sigma_{k}^{-1}(x-\mu_{k})-\frac{1}{2}\log|\Sigma_{k}|+\log\pi_{k}}}\\ {{}}&{{=}}&{{-\frac{1}{2}x^{T}\Sigma_{k}^{-1}x+x^{T}\Sigma_{k}^{-1}\mu_{k}-\frac{1}{2}\mu_{k}^{T}\Sigma_{k}^{-1}\mu_{k}-\frac{1}{2}\log|\Sigma_{k}|+\log\pi_{k}}}\end{array}$

因此，QDA类将${\Sigma}_k$，$\mu_k$ 和 $\pi_k$ 的估计值代入上式，然后将一个观测 $X = x$ 分配给这个量最大的类。与式[( 4.24 )](Classification%20de1ade1befa04bc79200cd723ad36f2f.md)不同的是，上式中数量 $x$ 以二次函数形式出现。这就是QDA得名的地方。

为什么我们假设 $K$ 类共享一个共同的协方差矩阵呢?换句话说，为什么人们更喜欢LDA而不是QDA，或者相反?

答案在于方差与偏差的权衡。有 $p$ 个预测变量时，估计一个协方差矩阵需要估计 $\frac{p(p+1)}{2}$ 个参数。QDA 为每个估计一个单独的协方差矩阵，共有 $Kp(p+1)/2$ 个参数。而 LDA 模型只用估计 $Kp$ 个线性系数。 因此，LDA是一个比QDA更不灵活的类，因此具有更低的方差。

而如果LDA对 $K$ 个类共享一个共同协方差矩阵的假设很差，那么LDA可以避免高偏差。粗略地说，在训练观测值相对较少的情况下，LDA往往比QDA更好，因此减小方差至关重要。相反，如果训练集非常大，以至于类的方差不是主要关注的问题，或者对于 $K$ 个类的公共协方差矩阵的假设显然是站不住脚的，则推荐使用QDA

### 朴素贝叶斯

> 朴素贝叶斯分类器对 $f_1(x),\dots,f_K(x)$ 采取不同的估计策略，我们没有假设这些函数属于特定的分布族(例如多元正态)，而是做了单一的假设：在第 $k$ 类中，$p$ 个预测变量是独立的。数学意义为：$k=1,\dots,K$$f_k(x)=f_{k1}(x_1)\times f_{k2}(x_2)\times\cdots\times f_{k p}(x_p)$ 
$f_{kj}$ 是第 $k$ 类观测值中第 $j$ 个预测因子的密度函数。
> 

本质上，估计一个 $p$ 维密度函数是很有挑战性的，因为我们不仅要考虑每个预测变量的边缘分布，即每个预测变量自身的分布，还要考虑预测变量的联合分布，即不同预测变量之间的相关性。

在多元正态分布的情况下，不同预测变量之间的关联由协方差矩阵的非对角线元素总结。

然而，一般来说，这种关联可能很难表征，而且很难估计。但通过假设  $p$ 个协变量在每个类内都是独立的，我们完全消除了担心 $p$ 个预测变量之间存在关联的需要，因为我们已经简单地假设了预测变量之间不存在关联

从本质上讲，朴素贝叶斯假设引入了一定的偏差，但减少了方差，由偏差-方差权衡，使得分类结果在实际中表现良好。

一旦我们做出了朴素贝叶斯假设，我们可以将( 4.29 )代入( 4.15 )得到后验概率的表达式:

$\Pr(Y=k|X=x)=\dfrac{\pi_k\times f_{k1}(x_1)\times f_{k2}(x_2)\times\cdots\times f_{kp}(x_p)}{\sum_{l=1}^K\pi_l\times f_{l1}(x_1)\times f_{l2}(x_2)\times\cdots\times f_p(x_p)}$

利用训练数据 $x_{1j},\dots,x_{nj}$ 估计一维密度函数 $f_{kj}$，我们有几个选择：

- 如果 $Xj$ 是定量的，那么我们可以假设 $X_j | Y = k\sim N( \mu_{jk} ,\sigma^2_{j k})$。也就是说，我们假设在每个类中，第 $j$ 个预测变量来自一个(单变量)正态分布。虽然这听起来有点像QDA，但有一个关键的区别，在这里我们假设预测变量是独立的；这相当于QDA增加了一个假设，即类别特异性协方差矩阵是对角的。
- 如果 $X_j$ 是定量的，那么另一种选择是对 $f_{kj}$ 使用非参数估计。一个非常简单的方法是对每个类中第 $j$ 个预测器的观测值做直方图。然后我们可以估计 $f_{kj} ( xj )$ 为第 $k$ 类中与 $x_j$ 属于同一直方图的训练观测值的分数。或者，我们可以使用核密度估计，它本质上是直方图的平滑版本。
- 如果 $X_j$ 是定性的，那么我们可以简单地统计每个类对应的第 $j$ 个预测器的训练观测值的比例。例如，假设 $X_j\in\{1,2,3\}$，第 $k$ 类共有100个观测值。假设第 $j$ 个预测因子在32、55和13个观测值中分别取值为1、2和3。那么我们可以估计 $f_{kj}$ 为
    
    $\hat{f}_{kj}(x_j)=\begin{cases}0.32&\text{if}x_j=1\\ 0.55&\text{if}x_j=2\\ 0.13&\text{if}x_j=3.\end{cases}$
    

我们期望在p较大或n较小的情况下，使用朴素贝叶斯相对于LDA或QDA有更大的收益，因此减小方差非常重要。

# 比较分类模型

### 分析比较

> 我们现在对LDA、QDA、朴素贝叶斯和逻辑回归进行分析(或数学)比较。我们在一个包含 $K$ 个类的集合中考虑这些方法，这样我们就可以将一个观测分配给最大化 $Pr( Y = k | X = x)$ 的类。
> 

首先，对于LDA，我们可以利用[贝叶斯定理( 4.15 )](Classification%20de1ade1befa04bc79200cd723ad36f2f.md)，并假设每个类内的预测变量来自一个具有特异类均值和共享协方差矩阵的多元正态密度( 4.23 )，以表明

$\begin{aligned}\log \left(\frac{\operatorname{Pr}(Y=k \mid X=x)}{\operatorname{Pr}(Y=K \mid X=x)}\right)= & \log \left(\frac{\pi_{k} f_{k}(x)}{\pi_{K} f_{K}(x)}\right) \\= & \log \left(\frac{\pi_{k} \exp \left(-\frac{1}{2}\left(x-\mu_{k}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(x-\mu_{k}\right)\right)}{\pi_{K} \exp \left(-\frac{1}{2}\left(x-\mu_{K}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(x-\mu_{K}\right)\right)}\right) \\= & \log \left(\frac{\pi_{k}}{\pi_{K}}\right)-\frac{1}{2}\left(x-\mu_{k}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(x-\mu_{k}\right) \\& +\frac{1}{2}\left(x-\mu_{K}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(x-\mu_{K}\right) \\= & \log \left(\frac{\pi_{k}}{\pi_{K}}\right)-\frac{1}{2}\left(\mu_{k}+\mu_{K}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mu_{k}-\mu_{K}\right) \\& +x^{T} \boldsymbol{\Sigma}^{-1}\left(\mu_{k}-\mu_{K}\right) \\= & a_{k}+\sum_{j=1}^{p} b_{k j} x_{j}\end{aligned}$$a_k=\log\left(\frac{\pi_k}{\pi_K}\right)-\frac{1}{2}(\mu_k+\mu_K)^T\Sigma^{-1}(\mu_k-\mu_K)$，$b_{kj}$ 是$\Sigma^{-1}(\mu_k-\mu_K)$ 的第 $j$ 个分量，因此，LDA与逻辑回归一样，假设后验概率的对数优势是关于 $x$ 的线性

通过类似计算，QDA可表明为

$\log\left(\dfrac{\Pr(Y=k|X=x)}{\Pr(Y=K|X=x)}\right)=a_k+\sum_{j=1}^p b_{kj}x_j+\sum_{j=1}^p\sum_{l=1}^p c_{kj}x_jx_l$

$a_k$，$b_{kj}$ 和 $c_{kjl}$ 是 $\pi_k,\pi_K,\mu_k,\mu_K,\Sigma_k,\Sigma_K$ 的函数。再次，顾名思义，QDA假设后验概率的对数优势比关于 $x$ 是二次的。

在朴素贝叶斯设定中，对于$j=1,\dots,p$ ，$f_k(x)$ 建模为 p 个一维函数的乘积 $f_{kj}(x_j)$，因此

$\begin{aligned}\log \left(\frac{\operatorname{Pr}(Y=k \mid X=x)}{\operatorname{Pr}(Y=K \mid X=x)}\right) & =\log \left(\frac{\pi_{k} f_{k}(x)}{\pi_{K} f_{K}(x)}\right) \\& =\log \left(\frac{\pi_{k} \prod_{j=1}^{p} f_{k j}\left(x_{j}\right)}{\pi_{K} \prod_{j=1}^{p} f_{K j}\left(x_{j}\right)}\right) \\& =\log \left(\frac{\pi_{k}}{\pi_{K}}\right)+\sum_{j=1}^{p} \log \left(\frac{f_{k j}\left(x_{j}\right)}{f_{K j}\left(x_{j}\right)}\right) \\& =a_{k}+\sum_{j=1}^{p} g_{k j}\left(x_{j}\right)\end{aligned}$

$a_k=log(\frac{\pi_k}{\pi_K})$ ，$g_{kj}(x_j)=log(\frac{f_{kj}(x_j)}{f_{Kj}(x_j)}$

通过对上述三式的检验，可以得到关于LDA、QDA和朴素贝叶斯的观察结果：

- LDA是QDA $c_{kjl} = 0$ 的特例，对所有 $j = 1,\dots,p$，$l = 1,\dots,p$，且$k = 1，\dots,K$  (当然，这并不奇怪，因为LDA只是QDA的一个限制版本,其中 $\Sigma_1=\cdots=\Sigma_K=\Sigma$ )
- 任何具有线性决策边界的分类都是 $g_{kj} ( xj ) = b_{kj}x_j$ 的朴素贝叶斯特例。特别地，这意味着LDA是朴素贝叶斯的一个特例！这一点在本章前面对LDA和朴素贝叶斯的描述中并不明显，因为每种方法都做了非常不同的假设：LDA假设特征服从正态分布且具有共同的类内协方差矩阵，而朴素贝叶斯则假设特征独立。
- 如果我们使用一维高斯分布 $N ( \mu_{kj} , \sigma^2_j)$ 对朴素贝叶斯类中的 $f_{kj} ( x_j )$ 进行建模，那么我们得到 $g_{kj} ( x_j ) = b_{kj} x_j$ ，其中$b_{kj} = ( \mu_{kj}-\mu{Kj})/\sigma^ 2_j$ 在这种情况下，朴素贝叶斯实际上是LDA的一个特例，其限制为第 $j$ 个对角元素等于 $\sigma^2_j$ 的对角矩阵。
- QDA和朴素贝叶斯都不是另一种情况的特例。朴素贝叶斯可以产生一个更灵活的拟合，因为对 $g_{kj} ( x_j )$ 可以做任何选择。然而，它仅限于一个单纯可加的拟合，即在[( 4.34 )](Classification%20de1ade1befa04bc79200cd723ad36f2f.md)式中，$x_j$ 的函数被添加到 $x_l$ 的函数中，因为 $j \neq l$；然而，这些项从来不是成倍增加的。相比之下，QDA包含形式为 $c_{kjl}x_jx_l$ 的乘法项。因此，在预测因子之间具有相关性时，QDA对区分类别有可能更准确。

这些方法中没有一种方法能统一地支配其他方法。在任何设定中，方法的选择将取决于每个 $K$ 类中预测变量的真实分布，以及其他考虑因素，如 $n$ 和 $p$ 的值。后者与偏差-方差的权衡有关。

逻辑回归是如何与这个故事联系在一起的？回想一下[( 4.12 )式](Classification%20de1ade1befa04bc79200cd723ad36f2f.md)，采用多项式逻辑回归形式：

$\log\left(\dfrac{\Pr(Y=k|X=x)}{\Pr(Y=K|X=x)}\right)=\beta_{k0}+\sum_{j=1}^p\beta_{kj}x_j$

这与LDA [( 4.32 )](Classification%20de1ade1befa04bc79200cd723ad36f2f.md)的线性形式相同：在两种情况下，$log ( \frac{Pr ( Y = k | X = x )}{ Pr ( Y = K | X = x )} )$ 都是预测变量的线性函数。在LDA中，这个线性函数中的系数是$\pi_k，\pi_K，\mu_k，\mu_K$ 的估计值的函数，并且通过假设 $X_1,\dots,X_p$ 在每个类内服从正态分布得到 $\Sigma$ 。相比之下，在逻辑回归中，选择系数是为了最大化似然函数[( 4.5 )](Classification%20de1ade1befa04bc79200cd723ad36f2f.md)。因此，当正态性假设(近似)成立时，我们预期LDA优于逻辑回归，而当正态性假设不成立时，我们预期逻辑回归表现更好。

回想 KNN （K-nearest neighbors）分类方法，为了对一个观测 $X = x$ 进行预测，找出与 $x$ 最接近的训练观测值。然后将 $X$ 分配给这些观测值的多个所属的类。因此，KNN是一种完全非参数方法：不对决策边界的形状做任何假设。我们对KNN做了以下观察：

- 由于KNN是完全非参数的，当决策边界高度非线性时，我们可以预期这种方法相较LDA和逻辑回归更有优势，只要 $n$ 非常大，$p$ 很小
- 为了提供准确的分类，相对于预测变量的数目，KNN 需要大量的观测值，即 $n$ 远大于 $p$。这与 KNN 是非参数的事实有关，在产生大量方差的同时，倾向于减小偏差
- 在决策边界为非线性但 $n$ 仅适中，或 $p$ 不太小的情况下，QDA可能优于KNN。这是因为QDA在利用参数形式优势的同时可以提供非线性的决策边界，这意味着相对于KNN，它可以用更小的样本容量进行准确的分类。
- 与逻辑回归不同，KNN 并没有告诉我们哪些预测因子是重要的，我们没有得到如下的系数表
    
    ![Untitled](Classification%20de1ade1befa04bc79200cd723ad36f2f/Untitled%202.png)
    

### 实证比较

书中例子说明，任何一种方法在任何情况下都不会占优势。当真实的决策边界是线性的，那么LDA和逻辑回归方法将倾向于表现良好。当边界为中度非线性时，QDA或朴素贝叶斯可能会给出更好的结果。最后，对于复杂得多的决策边界，KNN等非参数方法可能更有优势。但非参数方法的平滑程度必须谨慎选择。在下一章中，我们考察了选择正确的光滑度水平的一些方法，以及一般而言选择最佳的整体方法。

# 生成式线性模型

> 当响应变量即不是定量的，也不是定性的，之前的方法便都不适用了。
> 

### 在公共自行车数据上进行线性回归

如果在该数据集上使用线性回归，会有一些问题：

- 响应变量（用车人数）会为负数。由此，其余输出结果的准确性也会有问题
- 当骑行人数期望较小时，方差也应该较小。这严重违反了线性模型的假设，即 $Y =∑^p_{j=1} X_j \beta_j +\epsilon$ ，其中 $\epsilon$ 是方差为 $\sigma^2$ 的常数的均值为零的误差项，而不是协变量的函数。因此，数据的异方差性对线性回归模型的适用性提出了质疑。
- 最后，响应骑行者为整数值。但在线性模型下，$Y =\beta_0+∑^p_{j=1} X_j \beta_j +\epsilon$ ，其中 $\epsilon$ 为连续型误差项。这意味着在线性模型中，响应 $Y$ 必然是连续取值(定量)的。因此，响应骑行者的整数性质表明线性回归模型对该数据集并不完全令人满意。

<aside>
💡 转换响应变量避免了负预测的可能性，并且克服了未转换数据中的大部分异方差

$\log(Y)=\sum\limits_{j=1}^{p}X_j\beta_j+\epsilon$da

但如此会导致参数解释性的降低，且对数在相应为 0 的情况下无法转换

</aside>

### 对公共自行车数据进行泊松回归

> 泊松分布（Poisson distribution）假设随机变量Y取非负整数值，即 $Y∈\{ 0,1,2,\dots \}$ ，若 $Y$ 服从泊松分布，则
 $\Pr(Y=k)=\dfrac{e^{-\lambda}\lambda^k}{k!}\text{for}\space k=0,1,2,\ldots.$
$\lambda>0$ 是 $Y$ 的期望值 $E(Y)$ ，也等于 $Y$ 的方差，$\lambda=E(Y)=Var(Y)$
> 

这说明，如果 $Y$ 服从泊松分布，则 $Y$ 的均值越大，方差也越大。

通常采用泊松分布对计数进行建模；这是一个自然选择，原因有很多，包括计数像泊松分布一样取非负整数值。

我们考虑建立均值随协变量变化而变化的泊松分布。考虑如下模型  $\lambda=E(Y)$ ，记为 $\lambda(X_1,\dots,X_p)$ 来强调 $\lambda$ 是协变量 $X_1,\dots,X_p$ 的函数：

$\log(\lambda(X_1,\ldots,X_p))=\beta_0+\beta_1X_1+\cdots+\beta_pX_p$   或等价为：

$\lambda(X_1,\ldots,X_p)=e^{\beta_0+\beta_1X_1+\cdots+\beta_pX_p}$

$\beta_0,\beta_1,\dots,\beta_p$ 为待估参数

为了估计系数 $\beta_0,\beta_1,\dots,\beta_p$ ，我们采用与逻辑回归相同的最大似然法。具体地，给定泊松回归模型中的 $n$ 个独立观测值，似然函数为：

$\ell(\beta_0,\beta_1,\ldots,\beta_p)=\prod\limits_{i=1}^n\dfrac{e^{-\lambda(x_i)}\lambda(x_i)^{y_i}}{y_i!}$

其中$\lambda(x_i)=e^{\beta_0+\beta_1x_{i1}+\cdots+\beta_px_{ip}}$ 

泊松回归模型与线性回归模型的一些重要区别如下：

- 解释性：解释泊松回归的系数，必须密切关注 [公式](https://www.notion.so/Imagenet-classification-with-deep-convolutional-neural-networks-355a9503b14147259d620678c7db6711) ，表明 $X_j$ 增加一个单位，与由 $exp(\beta_j)$ 引起的 $E(Y)=\lambda$ 的改变有关。
- 均值方差关系：如前所述，在泊松模型下，$\lambda= E ( Y ) = Var ( Y )$ 。通过用泊松回归建模自行车使用情况，我们隐式地假设给定时间内的平均自行车使用量等于该时间内自行车使用量的方差。
- 非负的拟合值：通过 [公式](Classification%20de1ade1befa04bc79200cd723ad36f2f.md) 可看出，泊松模型本省只允许非负值，

### 更具一般性的广义线性模型

我们现在讨论了三类回归模型：线性、Logistic和Poisson。这些方法具有一些共同的特点：

- 每种方法都使用了预测变量 $X_1,\dots,X_p$ 来预测响应变量 $Y$ ,我们假设，$Y$ 属于某个确定的分布族。对于线性回归，我们通常假设Y服从高斯或正态分布。对于逻辑回归，假设Y服从伯努利分布。最后，对于泊松回归，我们假设Y服从泊松分布。
- 每种方法都将Y的均值建模为预测变量的函数
    - 线性模型中：
    $\operatorname{E}(Y|X_1,\ldots,X_p)=\beta_0+\beta_1X_1+\cdots+\beta_pX_p$
    - 逻辑回归中：
    $\begin{aligned} E(Y|X_1,\ldots,X_p) & = {Pr}(Y=1|X_1,\ldots,X_p)\\ & = \dfrac{e^{\beta_0+\beta_1X_1+\cdots+\beta_pX_p}}{1+e^{\beta_0+\beta_1X_1+\cdots+\beta_pX_p}} \end{aligned}$
    - 泊松分布中：
    $\operatorname{E}(Y|X_1,\ldots,X_p)=\lambda(X_1,\ldots,X_p)=e^{\beta_0+\beta_1X_1+\cdots+\beta_pX_p}$
    
    通过链接函数 $\eta$ 对$E(|YX_1,\dots,X_p)$ 转换，转换后的均值是预测变量的线性函数
    
    $\eta(\operatorname{E}(Y|X_1,\ldots,X_p))=\beta_0+\beta_1X_1+\cdots+\beta_pX_p$
    
    线性、Logistic和 Poisson 回归的连接函数分别为 $\eta(\mu)=\mu$，$\eta(\mu)=log(\mu/(1-\mu))$，$\eta(\mu)=log(\mu)$
    

高斯、伯努利和泊松分布都是更广泛的一类分布的成员，称为指数族。该族的其他著名成员是指数分布、Gamma分布和负二项分布。一般来说，我们可以通过将响应 $Y$ 建模为来自指数族中特定成员的回归，然后对响应的均值进行变换，使得变换后的均值通过( 4.42 )成为预测变量的线性函数。任何遵循这个非常普遍的公式的回归方法被称为广义线性模型( GLM )。因此，线性回归、逻辑回归和泊松回归是广义线性模型的三个例子。这里没有涉及的其他例子包括Gamma回归和负二项回归。