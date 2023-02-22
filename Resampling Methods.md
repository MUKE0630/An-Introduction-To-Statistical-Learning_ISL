# Resampling Methods

> 它们涉及从训练集中重复抽取样本，并在每个样本上重新拟合感兴趣的模型，以获得有关拟合模型的额外信息
> 

---

交叉验证可以用来估计与给定的统计学习方法相关的测试误差，以评估其性能，或者选择合适的灵活性水平。

<aside>
💡 评估模型性能的过程称为模型评估，而为模型选择合适的灵活性水平的过程称为模型选择

</aside>

自助法( bootstrap )在几种情况下被使用，最常见的是用来衡量参数估计的准确性或给定统计学习方法的准确性

# 交叉验证（*Cross-Validation*）

> 在没有非常大的指定测试集可以用来直接估计测试错误率的情况下，可以使用一些技术来使用可用的训练数据来估计这个数量。
> 

---

### 验证集方法

> 用于在一组观测值上估计特定统计学习方法的测试误差
> 

---

它涉及将可用观测集随机分为两部分，训练集（training set）和验证集（validation set）或保持集（hold-out set）模型在训练集上拟合，拟合好的模型在测试集上做预测。验证集的错误率提供了测试错误率的估计值。

验证集方法概念简单，易于实现。但它有两个潜在的弊端：

- 图下图所示，测试错误率的验证估计值可以是高度可变的，具体取决于哪些观测值包含在训练集中，哪些观测值包含在验证集中。

![Untitled](Resampling%20Methods%204b3811ead4184583a76913022f86d4bf/Untitled.png)

- 在验证方法中，只有观察的一个子集——那些包含在训练集中而不是验证集中的——被用来对模型进行测试。 由于统计方法在较少的观察值上训练时往往表现较差，这表明验证集错误率可能倾向于高估模型在整个数据集上的测试错误率。

---

### 留一交叉验证（*Leave-One-Out Cross-Validation*）

> 然而，与上节创建两个大小相当的子集不同，验证集使用单个观测值$( x1，y1)$，其余观测值 $\{ ( x_2 , y_2),\dots,( x_n , y_n) \}$ 构成训练集
> 

---

由于在拟合过程中没有使用 $( x_1 , y_1)$，$MSE_1 = ( y_1 -  \hat y_1)^2$ 为测试误差提供了近似无偏估计。但是，尽管MSE1对试验误差是无偏的，但由于它是基于单个观测值 $( x_1 , y_1)$ 的，所以它是一个很差的估计，因为它是高度可变的

对验证数据选择 $( x_2 , y_2)$，在 $n - 1$ 个观测值$\{ ( x_1 , y_1),( x_3 , y_3),\dots,( x_n , y_n) \}$，计算 $MSE_2 = ( y_2- \hat y_2 ) ^2$。重复这种做法 $n$ 次，产生 $n$ 个平方误差，$MSE_1,\dots,MSE_n$

LOOCV的测试 $MSE$ 估计是这 $n$ 个测试误差估计的平均值：

$$
\mathrm{CV}_{(n)}=\dfrac{1}{n}\sum_{i=1}^n\mathrm{MSE}_i\tag{5.1}
$$

与验证集方法相比，LOOCV具有几个主要优点：

- 具有更小的偏差。我们使用 $n-1$ 个观测值来拟合模型，几乎是整个数据集。因此，LOOCV方法不像验证集方法那样高估测试错误率。
- 验证集方法划分训练/验证集时的随机性，重复应用时会导致不同的结果；LOOCV方法总会得到相同的结果，因为训练/验证集的划分不存在随机性

LOOCV 方法的应用成本可能很高，因为一个模型需要拟合 $n$ 次，特别是 $n$ 特别大 或 模型拟合慢的时候。这里有一个捷径使最小二乘与多项式回归具有相同的代价：

$$
\mathrm{CV}_{(n)}=\dfrac{1}{n}\sum_{i=1}^n\left(\dfrac{y_i-\hat{y}_i}{1-h_i}\right)^2\tag{5.2}
$$

$\hat y_i$ 是原始最小二乘法拟合中的第 $i$ 个响应值，$h_i$ 是 [式( 3.37 )](Linear%20Regression%20a99b01f6ea794d299cbfbfa76cf73d36.md) 的杠杆量。这与普通的 $MSE$ 类似，只是第 $i$ 个残差除以 $1 - h_i$ 。杠杆量介于 $\frac1n$ 和 1 之间，反映了观测量对自身拟合的影响程度。因此，在这个公式中，高杠杆点的残差恰好是这个等式成立的合适量。这个公式普遍并不成立，在这种情况下，模型必须重新拟合 $n$  次

### K折交叉验证（*k-Fold Cross-Validation*）

> 将数据集随机分为 $k$ 组（称为 $k$ 折），大小近似相等。第一折作为验证集，剩余作为训练集，在验证集上得到 $MSE_1$ 。这个过程重复 $k$ 次，每次选择不同的验证集，得到 $k$  个测试集的估计值 $MSE_1,\dots,MSE_k$
> 

$k$ 重CV 估计的平均值：

$$
\mathrm{CV}_{(k)}=\dfrac{1}{k}\sum_{i=1}^kMSE_i\tag{5.3}
$$

$k$-折交叉验证是 LOOCV 的特例。其最大优点在于计算量，$k=n$ 时需要模型进行 n 次拟合，而 k 折只需进行 k 次拟合。而执行 $k=5$ 和 $k=10$ 可能存在非计算上的优势，这涉及到偏差-方差权衡

当我们进行交叉验证时，我们的目标可能是确定**给定的统计学习**过程在独立数据上的表现；在这种情况下，测试 $MSE$ 的实际估计是有意义的。

但在其他时候我们只对估计的测试 $MSE$ 曲线中的**最小值点的位置**感兴趣。这是因为我们可能在多个统计学习方法上进行交叉验证，或者在单个方法上使用不同的灵活性水平，以识别导致最低测试误差的方法。为此，估计的测试MSE曲线中最小点的位置很重要，但估计的测试MSE的实际值并不重要。

### $k$ 折交叉验证的偏差-方差权衡

> K-折交叉验证的一个不太明显但潜在更重要的优点是它经常给出比LOOCV更精确的测试错误率估计。这与偏差-方差权衡有关
> 

从降低偏差角度来看，LOOCV 要优于 K-折交叉验证。

因为 LOOCV 使用了几乎（ $n-1$ 个观测值）整个数据集，而 K-折交叉验证每个训练集包含 $(k-1)n/k$ 个观测值

但 K-折交叉验证 比 LOOCV 具有更低的方差。

当我们执行 LOOCV 时，我们实际上是对 $n$ 个拟合模型的输出进行平均，每个模型都在几乎相同的观察集上进行训练， 因此，这些输出彼此高度（正）相关。

相比之下，当我们执行 $k < n$ 的 k 折 CV 时，我们正在对彼此相关性较低的 $k$ 个模型的输出进行平均，因为每个模型中训练集之间的重叠较小。

由于许多高度相关量的均值比许多不太相关量的均值具有更高的方差，因此LOOCV得到的试验误差估计值往往比K-折交叉验证得到的试验误差估计值具有更高的方差。

综上，在 K 折交叉验证中，存在一个与 k 的选择相关的偏差-方差权衡。通常，考虑到这些因素，我们使用 $k = 5$ 或 $k = 10$ 进行 k-折交叉验证，因为这些值已根据经验显示产生测试错误率估计，既不会受到过高的偏差也不会受到非常高的方差的影响       

### 分类问题的交叉验证

> 不使用 $MSE$ 来量化测试误差，而是用错误分类的观测数
> 

例如，在分类设置中，LOOCV错误率取形式

$$
\mathrm{CV}_{(n)}=\dfrac{1}{n}\sum_{i=1}^n\mathrm{Err}_i \tag{5.4}
$$

${Err}_i=I(y_i\neq\hat y_i)$，k-折 CV错误率和验证集的错误率定义类似

![图 5.8 中显示的二维分类数据的测试误差（棕色）、训练误差（蓝色）和 10-折 CV 误差（黑色）。 左：使用预测变量的多项式函数的逻辑回归。 所用多项式的顺序显示在 x 轴上。 右图：具有不同 K 值的 KNN 分类器，KNN 分类器中使用的邻居数量。](Resampling%20Methods%204b3811ead4184583a76913022f86d4bf/Untitled%201.png)

图 5.8 中显示的二维分类数据的测试误差（棕色）、训练误差（蓝色）和 10-折 CV 误差（黑色）。 左：使用预测变量的多项式函数的逻辑回归。 所用多项式的顺序显示在 x 轴上。 右图：具有不同 K 值的 KNN 分类器，KNN 分类器中使用的邻居数量。

# 自助法（*Bootstrap*）

> 自助法( bootstrap )是一种应用广泛且极其强大的统计工具，可以用来量化与给定估计量或统计学习方法相关的不确定性
> 

Bootstrap方法允许我们使用计算机模拟获得新样本集的过程，这样我们就可以在不产生额外样本的情况下估计 $\hat{\alpha}$ 的变异性

我们不是从总体中重复获得独立的数据集，而是**通过从原始数据集中重复采样观测值来获得不同的数据集。** 如下图

![图5.11 包含 $n = 3$ 个观测值的小样本的 bootstrap 方法的图形说明。 每个 bootstrap 数据集包含 $n$ 个观测值，从原始数据集中进行替换采样。 每个 bootstrap 数据集用于获得 $\alpha$ 的估计值。](Resampling%20Methods%204b3811ead4184583a76913022f86d4bf/Untitled%202.png)

图5.11 包含 $n = 3$ 个观测值的小样本的 bootstrap 方法的图形说明。 每个 bootstrap 数据集包含 $n$ 个观测值，从原始数据集中进行替换采样。 每个 bootstrap 数据集用于获得 $\alpha$ 的估计值。

采样是有放回的，意味着相同的观测值可以多次出现在一个 bootstrap 数据集中。产生 B 个bootstrap 集 $Z^{*1},\dots,Z^{*B}$，同时产生 B 个估计值 $\hat \alpha^{*1},\dots,\hat\alpha^{*B}$，计算 bootstrap 估计的标准误差

$$
\operatorname{SE}_B(\hat\alpha)=\sqrt{\dfrac{1}{B-1}\sum_{r=1}^B\left(\hat\alpha^{*r}-\dfrac{1}{B}\sum_{r'=1}^B\hat\alpha^{*r'}\right)^2}\tag{5.8}
$$