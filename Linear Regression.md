# Linear Regression

# 简单线性回归

> 简单线性回归形式：$Y=\beta_0+\beta_1X_1$
> 

### 估计系数

残差：$e_{i}=y_{i}-\hat{y}_{i}$

残差平方和（*residual sum of squares* RSS）$RSS=\sum_{i=1}^{n}e_i^2$   用最小二乘法近似 $\hat\beta_0$ 和 $\hat\beta_1$ 来降低 RSS，得出极小值点

 $\hat{\beta}_{1}=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}$        $\hat{\beta}_{0}=\bar{y}-\hat{\beta}_{1} \bar{x}$

### 评估系数精准度

$Y$ 与 $X$ 的线性关系写成 $Y = \beta_0 +\beta_1X+\epsilon$

无偏估计量不会系统的高估或低估真实参数。无偏估计量依赖于数据集，在特定数据集上会高估，通过大量数据集估计后再平均会更加精确

- 对样本均值 $\mu$ 的估计后得到 $\hat\mu$ ，如何评估估计量 $\hat{\mu}$ 的偏离？
    - 使用标准误差（*standard error*，SE）$Var(\hat\mu)=SE(\hat\mu)=\frac{\sigma^2}{n}$ 其中 $\sigma$ 是每个 $y_i$ 与 $Y$ 的标准差  （ n 个观测值是不相关的）
    - 相同的，我们可以及计算 $\hat\beta_0$ 和 $\hat\beta_1$ 的标准差来评估两者与真值的接近程度，$\operatorname{SE}\left(\hat{\beta}_{0}\right)^{2}=\sigma^{2}\left[\frac{1}{n}+\frac{\bar{x}^{2}}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}\right]$， $\operatorname{SE}\left(\hat{\beta}_{1}\right)^{2}=\frac{\sigma^{2}}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}$
    其中 $\sigma^2=Var(\epsilon)$，每个误差项有共同的方差且互不相关。当 $x_i$ 越分散，$SE(\hat\beta_1)$ 越小；当 $\bar{x}$ 为零 $SE(\hat\beta_0)$ 与$SE(\hat\mu)$ 相等
    - 一般而言，$\sigma^2$ 是不知道的，但是可以由数据来估计，这个估计值称为残差标准差（residual standard error，RSE）$RSE=\sqrt{(RSS/(n-2)}$
    - 标准误差可以用于计算置信区间（confidence interval）在一个置信度下（95%）定义一个取值范围 [a , b]，使得在该置信度下，该范围包含真实的未知值。
- 对于线性回归，95%置信区间近似取为 $\hat\beta_1\space\pm\space2*SE(\hat\beta_1)$ 
区间为 $\left[\hat{\beta}_{1}-2 \cdot \operatorname{SE}\left(\hat{\beta}_{1}\right), \hat{\beta}_{1}+2 \cdot \operatorname{SE}\left(\hat{\beta}_{1}\right)\right]$
- 标准差可以用于对系数进行假设检验（hypothesis test）
    - 例如，对于原假设 $H_0:\beta_1=0$ ，如果 $SE(\hat\beta_1)$ 小，即使 $\hat\beta_1$ 很小，也可以拒绝原假设；相反，如果 $SE(\hat\beta_1)$  较大，那么 $\hat\beta_1$ 必须很大才能拒绝原假设。实践中，我们计算 t - 统计量（t - statistic）$t=\frac{\hat\beta_1-0}{SE(\hat\beta_1)}$ ，该数值量化了 $\hat\beta_1$ 标准差对 0 的远离；如果 $X$ 与 $Y$ 没有关系，统计量将会有 $n-2$ 个自由度的 $t$  分布。绝对值大于等于 $|t|$ 的数的概率称为 $p$值（*p-value*）一个较小的 p 值表明，在 $X$ 与 $Y$ 之间不存在任何实际关联情况下，不太可能存在偶然的原因，观测到  $$ $X$ 与 $Y$ 之间存在实际的联系；因此，如果观测到很小的 p 值可以推断 $X$ 与 $Y$ 之间存在关联。典型 p 值范围是 5% 或 1%

### 评估模型精确性

线性回归通常使用两个量来评估模型：残差标准差（residual standard error，RSE）和 $R^2$ 统计量

- 残差标准差
    
    > RSE是对 $\epsilon$ 标准差的估计。粗略说，是响应变量偏离真实回归线的平均值
    > 
    
    $\mathrm{RSE}=\sqrt{\frac{1}{n-2} \mathrm{RSS}}=\sqrt{\frac{1}{n-2} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}}$   
    
    $RSS=\sum_{i=1}^n(y_i-\hat{y_i})^2$
    
- $R^2$ 统计量
    
    $R^2=\frac{TSS-RSS}{TSS}=1-\frac{RSS}{TSS}$    $TSS=\sum(y_i-\bar{y})^2$
    
    - TSS 测量响应 Y 中的总方差，可以将其视为执行回归之前响应中固有的变异量。 相比之下，RSS 测量执行回归后无法解释的变异量。 因此，TSS − RSS 测量通过执行回归可以被解释的变异量，R2 测量 Y 中可以使用 X 解释的变异量比例。
    - 确定一个好的 $R^2$ 需要根据应用情况，在确定的物理模型中，我们希望 $R^2$ 的值尽可能大，而在只能粗略近似估计的情况下，例如生物，心理，市场营销等模型中，我们预计只有很小一部分的响应方差可以由预测变量解释，此时更低的 $R^2$ 可能更符合实际
    - $R^2$ 统计量是 $X$ 和 $Y$之间线性关系的度量。相关性（*correlation*）定义为
     $\operatorname{Cor}(X, Y)=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sqrt{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}} \sqrt{\sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)^{2}}}$ 
    所以我们用 $r=Cor(X,Y)$ 代替 $R^2$ ，**在简单线性回归中**， $R^2=r^2$
        - 相关性量化是单变量对单变量之间的关联，而不是多变量之间的关联

# 多元线性回归

> 多元线性回归形式：$Y=\beta_0+\beta_2X_1+....+\beta_pX_P+\epsilon$
> 

### 评估回归系数

### 一些重要的问题

1. 至少有一个 $X$ 对预测有用？
2. 所有预测变量都有助于解释响应变量，或者仅是一个子集的预测变量有用？
3. 模型拟合数据的效果如何
4. 给定一个预测变量集合，我们会预测出怎样的响应变量，预测的精度如何？

- 预测变量和响应变量之间有关系吗？
    
    使用假设检验来验证这个问题，$H_0:\beta_1=\beta_2=...=\beta_p=0$
    $H_\alpha:{\exists}\space\beta_j\neq0$  通过计算 $F-statistic$ 来论证
    $F=\frac{TSS-RSS)/P}{RSS/(N-P-1)}$    $TSS=\sum(y_i-\bar{y})^2$    $RSS=\sum_{i=1}^n(y_i-\hat{y_i})^2$
    
    如果线性假设是正确的，则 $E\{\operatorname{RSS} /(n-p-1)\}=\sigma^{2}$
    
    如果假设 $H_0$ 是正确的，则
    
    因此当响应变量与预测变量之间不存在关联时，我们预计 F 统计量的取值接近于1，如果 $H_\alpha$ 为真，则预计 F 大于1
    
    - 当 F 统计量已经接近1，那么拒绝 $H_0$ 需要多大的 F ？这取决于 n 和 p 的取值
        
        对任意给定的 n 和 p 可以计算 p 值（p - value）从而判断是否拒绝 $H_0$
        

- 确定重要变量
    - 前向选择，从零（预测）变量模型开始，加入能导致 RSS 最低的变量，知道满足某种规则后停止。
    - 后向选择，从模型中所有变量开始，删除 p 值 最大的变量，即删除统计显著性最小的变量。知道满足某种规则停止。
    - 混合选择，一开始使用前向选择，随着预测变量的加入，变量的 p 值会增大。如果模型中任何一个变量的 p 值在加入某个点后超过了阈值，则从模型中删除该变量。知道模型中所有变量都有一个很低的 p 值，而模型外的变量加入到模型中就会有一个很大的 p 值。
    
    如果p > n，则不能使用后向选择，而总是可以使用前向选择。前向选择是一种贪婪的方法，并且可能包含早期的变量，后来变得冗余。混合选择可以弥补这一点。
    
- 模型拟合
    - 当模型中加入更多的变量时，即使这些变量与响应只有微弱的关联，$R^2$ 也会增加。这是由于增加另一个变量总是导致训练数据（尽管不一定是测试数据）上的残差平方和减小。因此，同样在训练数据上计算的 $R^2$ 统计量必须增加。
    - $RSE=\sqrt{\frac{RSS}{n-p-1}}$ 
    当RSS的减小相对于p的增大较小时，含有较多变量的模型可以具有较高的 RSE
    - 图形摘要可以揭示从数值统计中不可见的模型的问题

- 预测
    
    预测有三种不确定性
    
    - 系数估计的不准确性与可约误差（reducible error）有关，可以通过计算置信区间来确定 $\hat{Y}$ 与 $f(X)$ 的接近程度
    - 假设 $f(X)$ 为线性模型是一种对现实的近似，因此引入了一个额外的可减少误差，称之为模型偏差（model bias）但当我们使用线性模型时，我们将忽略这个差异
    - 预测区间总是比置信区间宽，因为其既包含了对 $f(X)$ 估计的误差（可约误差）和个体点偏离回归平面的不确定性（不可约误差）

# 其他对线性回归的考虑

### 定性预测变量

- 只有两个水平的预测变量
    
    创建虚拟变量 （在机器学习中称为 独热编码）
    
    $x_{i}=\left\{\begin{array}{ll}1 & \text { if } i \text { th person owns a house } \\0 & \text { if } i \text { th person does not own a house }\end{array}\right.$
    在回归方程中使用该变量作为预测因子
    
    $y_{i}=\beta_{0}+\beta_{1} x_{i}+\epsilon_{i}=\left\{\begin{array}{ll}\beta_{0}+\beta_{1}+\epsilon_{i} & \text { if } i \text { th person owns a house } \\\beta_{0}+\epsilon_{i} & \text { if } i \text { th person does not. }\end{array}\right.$
    
    $\beta_0$ 解释为不拥有信用卡的人之间的平均信用卡余额
    
    或者使用别的虚拟变量代替 0/1 编码方案
    
    $x_{i}=\left\{\begin{array}{ll}1 & \text { if } i \text { th person owns a house } \\-1 & \text { if } i \text { th person does not own a house }\end{array}\right.$
    
    则对应该回归方程为
    
    $y_{i}=\beta_{0}+\beta_{1} x_{i}+\epsilon_{i}=\left\{\begin{array}{ll}\beta_{0}+\beta_{1}+\epsilon_{i} & \text { if } i \text { th person owns a house } \\\beta_{0}-\beta_{1}+\epsilon_{i} & \text { if } i \text { th person does not. }\end{array}\right.$
    
    $\beta_0$ 解释为总体平均信用卡余额（忽略住房影响）
    
    值得注意的是，无论使用何种编码方案，所有者和非所有者的信用余额的**最终预测都将是相同的，唯一的区别在于系数的解释方式。**
    
- 超过两个水平的定性预测变量
    
    创建额外虚拟变量（例：对地区创建虚拟变量）
    
    $x_{i 1}=\left\{\begin{array}{ll}1 & \text { if } i \text { th person is from the South } \\0 & \text { if } i \text { th person is not from the South }\end{array}\right.$
    
    $x_{i 2}=\left\{\begin{array}{ll}1 & \text { if } i \text { th person is from the West } \\0 & \text { if } i \text { th person is not from the West }\end{array}\right.$
    
    建立线性回归方程
    
    $y_{i}=\beta_{0}+\beta_{1} x_{i 1}+\beta_{2} x_{i 2}+\epsilon_{i}=\left\{\begin{array}{ll}\beta_{0}+\beta_{1}+\epsilon_{i} & \text { if } i \text { th person is from the South } \\\beta_{0}+\beta_{2}+\epsilon_{i} & \text { if } i \text { th person is from the West } \\\beta_{0}+\epsilon_{i} & \text { if } i \text { th person is from the East. }\end{array}\right.$
    

### 线性模型的扩展

> 线性模型做出了几个高度限制性的假设，其中最重要的两个假设是预测变量和响应变量之间的关系是**可加的**和**线性的**。
> 

可加性假设意味着一个预测变量 $X_j$ 和响应变量 $Y$  之间的相关性不依赖于其他预测变量的值

线性假设指出，与  $X_j$ 的一个单位变化相关的响应 $Y$ 的变化是恒定的，不管 $X_j$ 的值如何

- 移除可加性假设
    - 扩展模型的一种方法是包含第三个预测因子，称为交互项，它是通过计算 $X_1$ 和 $X_2$ 的乘积构造的
    $Y=\beta_{0}+\beta_{1} X_{1}+\beta_{2} X_{2}+\beta_{3} X_{1} X_{2}+\epsilon$
        
        交互项如何减轻可加性假设？
        
        $\begin{aligned}Y & =\beta_{0}+\left(\beta_{1}+\beta_{3} X_{2}\right) X_{1}+\beta_{2} X_{2}+\epsilon \\& =\beta_{0}+\tilde{\beta}_{1} X_{1}+\beta_{2} X_{2}+\epsilon\end{aligned}$
        
        $\tilde{\beta_1}=\beta_1+\beta_3X_2$ ，因此 $\tilde{\beta_1}$ 是 $X_2$ 的函数。$X_1$ 和 $Y$ 之间的关联不再是恒定的，$X_2$ 值的变化会改变 $X_1$ 和 $Y$ 之间的关联
        
        如果 $X_1$  和 $X_2$ 之间的相互作用看起来很重要，那么即使 $X_1$  和 $X_2$ 的系数估计值具有较大的 p 值，我们也应该将它们都包括在模型中。因为，如果 $X_1 × X_2$ 与响应变量有关，那么 $X_1$ 或 $X_2$ 的系数是否恰好为零就没有什么意义了。
        

- 非线性关系
    
    例如假设关系为：$Y=\beta_0+\beta_1\times X_1+\beta_2\times {X_2}^2+\epsilon$ia
    

### 潜在问题

- 数据的非线性
    
    > 利用残差图识别非线性
    > 
    
    给定一个简单线性回归模型，将残差 $e_i=y_i-\hat{y_i}$ 与预测变量 $x_i$ 作图。理想情况下，残差图将显示无可辨别的模式。 模式的存在可能表明线性模型的某些方面存在问题
    
    ![Untitled](Linear%20Regression%20a99b01f6ea794d299cbfbfa76cf73d36/Untitled.png)
    
    左图 U 型图指示了数据中强烈的非线性，右图无明显的模式
    
- 误差项的相关性
    
    > 线性回归模型的一个重要假设是误差项不相关
    > 
    
    如果误差项之间具有相关性：
    
    则估计值的标准误差将低于真实标准误差，会导致置信区间与预测区间会比应有范围更窄
    
    与模型相关的 p 值会低于他们应有的值，这可能导致我们错误地认为某个参数具有统计显著性
    
    假设我们不小心把我们的数据加倍，导致观测值和误差项两两相同。如果我们忽略这一点，我们的标准误差计算就好像我们有一个大小为 $2n$ 的样本，而实际上我们只有 $n$  个样本。我们对 $2n$ 个样本的估计参数与对 $n$ 个样本的估计参数相同，但置信区间缩小了$\sqrt2$ 
    
- 误差项的非恒定方差
    
    > 线性回归模型的另一个重要假设是 误差项具有恒定的方差 $Var(\epsilon_i)=\sigma^2$ 与线性模型相关的标准误差、置信区间、假设检验都依赖于这一假设
    > 
    
    但误差项的方差往往是非常数的。我们可以通过残差图中漏斗形状的存在来识别误差中的非恒定方差或异方差性。
    
- 异常值（离群点）
    
    > 离群点是指 $y_i$ 离模型的预测值较远的点
    > 
    
    离群值对于模型的拟合可能不会有太大的影响，但会引入其他问题。RSE 可能会因为单个离群值而发生急剧的变化，而 RSE 用于计算所有的置信区间和 p 值，因此单个数据引起的这种变化，可能会对拟合的解释产生影响。同样离群值的加入会导致 $R^2$ 的变化
    
    残差图可以用于识别异常值，但很难确定残差需要多大。可以画学生残差（studentized residuals）用每个残差除以它的估计标准差，学生化残差绝对值大于 3 的观测值可能是异常值
    
    因为数据收集错误的异常值可以直接删除，但需要注意，异常值也可以用于指示模型的不足
    
- 高杠杆点（High Leverage Points）
    
    > 不寻常的预测变量 $x_i$
    > 
    
    在简单线性回归中，高杠杆值一般是超过正常值的
    
    在多元线性回归中，预测值在单个预测因子中是良好的，但在全集中是不正常的
    
    为了量化观测值的杠杆，我们计算杠杆统计量（leverage statistic）该统计量的值较大，表明观测值具有较高的杠杆统计杠杆。
    $h_{i}=\frac{1}{n}+\frac{\left(x_{i}-\bar{x}\right)^{2}}{\sum_{i^{\prime}=1}^{n}\left(x_{i^{\prime}}-\bar{x}\right)^{2}}$ 
    杠杆统计量 $h_i$ 始终介于 $\frac1n$ 和 $1$ 之间，所有观测值的平均杠杆始终等于 $\frac{p + 1}{n}$。因此，如果一个给定的观测有一个远远超过  $\frac{p + 1 }{n}$ 的杠杆统计量，那么我们可能会怀疑对应点有高杠杆。
    

- 共线性（collinearity）
    
    > 共线性是指两个或多个预测变量密切相关的情况
    > 
    
    共线性的存在会在回归背景下带来问题，因为很难分离出共线变量对响应的单独影响
    
    共线性降低了回归系数估计值的准确度，导致 $\hat{\beta_j}$ 的标准误差增大。而 t-统计量由每个 $\hat{\beta_j}$ 除以其标准误差计算得到，因此共线性导致 t-统计量下降。因此，当存在共线性问题时，我们可能无法拒绝  $H_0:\beta_j=0$ ，这意味着共线性降低了假设检验的能力，正确检测非零系数的概率。 
    
    评估多重共线性的一个更好方法是计算方差膨胀因子（VIF）VIF的最小可能值为1，表明完全不存在共线性。实践中一般存在少量共线性，当 VIF 超过 5 或 10 时表明存在共线性问题。VIF公式为：
    $\operatorname{VIF}\left(\hat{\beta}_{j}\right)=\frac{1}{1-R_{X_{j} \mid X_{-j}}^{2}}$    
    
    ${R_{X_{j} \mid X_{-j}}^{2}}$是 $X_j$ 对所有其他预测变量回归的 $R_2$
    
    解决共线性的两个办法：
    
    1. 从回归中删除其中一个有问题的变量。这通常可以在不影响回归t的情况下完成，因为共线性的存在意味着该变量提供的关于响应的信息在其他变量存在的情况下是冗余的
    2. 将共线变量组合成单个预测变量。

- 一些问题
    1. 预测变量和响应变量之间是否存在关联？
        
        做出 $H_0：\beta=0$ 后，F-统计量可以判断是否拒绝这种假设
        
    2. 这种关联有多强？
        
        RSE 用于估计响应变量的标准差；$R^2$ 记录由预测因子解释的响应中变异的百分比。
        
    3. 哪个预测变量与响应相关？
        
        检查与每个预测因子的t统计量相关的p值
        
    4. 每个预测变量与响应俩之间的关联有多大？
        
        置信区间。但报纸的区间包含0，说明给定电视和广播的取值，该变量在统计上不显著
        
    5. 预测的响应变量准确度多高？
        
        与此估计相关的准确性取决于我们是否希望预测个体响应$Y = f ( X ) +\epsilon$ ，或平均响应 $f ( X )$ 。如果是前者，我们使用预测区间，如果是后者，我们使用置信区间。预测区间总是比置信区间宽，因为它们考虑了与不可约误差相关的不确定性。
        
    6. 这种关联是线性的吗？
        
        使用残差图识别非线性
        

### 对比线性回归和 K最近邻法