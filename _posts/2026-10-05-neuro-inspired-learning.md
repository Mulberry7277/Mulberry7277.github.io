---
title: "神经启发的学习机制：从局部突触可塑性到持续学习"
date: 2026-09-28 18:30:00 +0800
categories: [学术前沿, 论文解读]
tags: [神经启发学习, 突触可塑性, Hebbian learning, STDP, 三因子学习规则, e-prop, 树突计算, 持续学习, 脉冲神经网络, 数字生命]
math: true
---

> **文章状态：持续更新**
>
> 本文讨论“神经元和突触怎样学习”，重点放在局部突触可塑性、奖励调制、时间信用分配、树突计算和持续学习。论文不按期刊等级排列，而是按它们在机制链条中的作用组织：生物机制提供问题，算法论文给出实现，数字生命模型检验这些规则能否进入具身闭环。

## 1. 先区分三件事：网络动力学、连接结构和学习规则

神经启发人工智能常常同时讨论脉冲神经元、树突、连接组和突触可塑性，但它们回答的是不同问题：

| 层次 | 主要问题 | 常见例子 |
| --- | --- | --- |
| 神经动力学 | 神经元如何随时间产生电位和脉冲？ | LIF、ALIF、树突时间常数 |
| 网络结构 | 哪些神经元与哪些神经元相连？ | 局部连接、侧向抑制、果蝇连接组 |
| 学习机制 | 经验到来后，哪些参数如何改变？ | Hebbian、STDP、三因子规则、e-prop |

一个网络可以使用脉冲神经元，却用标准反向传播训练；也可以使用连续值神经元，却用局部 Hebbian 规则学习。因此，“用了脉冲”不等于“学习机制已经具有生物可实现性”。本文主要关心第三层，同时说明它依赖前两层。

在深度学习中，反向传播和循环网络中的 BPTT 已经非常有效。它们在生物实现上通常面临四个问题：

- 更新一个突触时，往往需要远处神经元的误差信号；
- 训练循环网络时，需要保存或重建较长时间内的网络状态；
- 前向权重和反向传播权重之间存在权重传输问题；
- 学习信号常常是批量、离线和全局的，而动物可以在行为过程中持续学习。

所以，神经启发学习机制真正想回答的是：**如果一个突触主要只能看到自己两端的活动，再接收少量来自神经调质、反馈或奖励的信号，它能否完成有用的信用分配？**

本文将这条问题链拆成五步：局部活动留下迹线，调制信号筛选迹线，反馈或树突提供局部目标，持续学习机制保护旧知识，最后把这些规则放入身体—环境闭环。

## 2. 一个最小的局部学习网络

在阅读复杂模型前，可以先把问题压缩成一个最小网络：输入层编码脉冲，中间层使用 LIF 神经元，突触通过 STDP 或三因子规则更新，侧向抑制让神经元竞争有限的表征资源。

### 2.1 LIF 神经元和脉冲活动

一个离散时间的漏电积分发放神经元可以写成：

$$
u_j[t+1]=\lambda u_j[t]+\sum_i w_{ij}s_i[t]+I_j[t],
$$

$$
s_j[t]=H\left(u_j[t]-\vartheta_j\right),
$$

其中 $u_j[t]$ 是膜电位，$s_i[t]\in\{0,1\}$ 是突触前脉冲，$\lambda$ 是漏电系数，$\vartheta_j$ 是阈值，$H(\cdot)$ 是阶跃函数。发放后可以采用软复位：

$$
u_j[t]\leftarrow u_j[t]-\vartheta_j s_j[t].
$$

这里的 LIF 只是神经动力学模型，不包含学习规则。学习规则决定的是 $w_{ij}$ 如何改变。

### 2.2 STDP 网络中的竞争机制

如果所有神经元都按照相关性增强，权重容易一起变大，神经元也可能学到几乎相同的模式。实际的 SNN 通常会加入：

- 侧向抑制：获胜神经元抑制同层其他神经元；
- 自适应阈值：经常发放的神经元阈值上升；
- 权重归一化或上下界：防止 Hebbian 增长失控；
- 稀疏发放：让不同神经元分工表示不同模式。

Diehl 和 Cook 的工作是一个很适合入门的完整实例：网络使用电导型突触、STDP、侧向抑制和自适应阈值，在没有标签的情况下学习数字表征。[论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4522567/)

一个简化的局部更新流程如下：

```text
for each time step t:
    read input spikes s_i[t]
    update every LIF membrane potential
    emit output spikes s_j[t]
    update pre/post traces from s_i[t], s_j[t]
    if neuron j wins the lateral competition:
        update only synapses connected to j
    clip or normalize each updated weight
```

这个例子很重要，因为它把“局部规则”落实成了可运行的网络结构，而不是只停留在一条抽象公式上。

## 3. Hebbian learning 和 STDP：局部活动如何改变突触

### 3.1 Hebbian learning：相关活动是候选改变

最简单的 Hebbian 规则为：

$$
\Delta w_{ij}=\eta x_i y_j,
$$

其中 $x_i$ 和 $y_j$ 分别表示突触前、突触后活动。它具有很强的局部性：突触只需要读取自己两端的活动，不需要知道网络的全局损失。

但原始规则会导致权重不断增长，也没有说明这次相关活动是否对任务有利。因此，实际模型通常需要归一化、抑制、权重边界或稳态调节。Hebbian learning 更适合作为“相关性写入”的基本思想，而不是完整的学习系统。

### 3.2 STDP：把时间顺序加入 Hebbian 规则

脉冲时间依赖可塑性（spike-timing-dependent plasticity，STDP）把前、后脉冲的时间差加入 Hebbian 学习。如果突触前脉冲先到、突触后神经元随后发放，连接通常增强；如果顺序相反，连接通常减弱。一个常见的指数窗口是：

$$
\Delta w =
\begin{cases}
A_+\exp(-\Delta t/\tau_+), & \Delta t>0,\\
-A_-\exp(\Delta t/\tau_-), & \Delta t<0,
\end{cases}
$$

其中 $\Delta t=t_{\mathrm{post}}-t_{\mathrm{pre}}$。具体方向和时间常数会随脑区、受体、膜电位和神经调质而变化，所以这条曲线应理解为模型中的典型形式。

![STDP 的典型时间窗：突触前脉冲早于突触后脉冲时，权重变化常为正；顺序相反时，常为负](/assets/img/posts/neuro-learning-stdp.svg)

*图 1。STDP 的典型时间窗示意。横轴为突触前后脉冲的时间差，纵轴为突触权重变化；具体窗口依赖神经回路和实验条件。本文自绘。*

STDP 的核心优点是局部性和时间敏感性。它能够强化“哪些输入先出现、哪些输入随后导致了发放”的关系，因此可以学习时序模式。它的核心局限是：单靠前、后脉冲相关性，它并不知道这次活动是否带来了奖励，也很难处理几秒之后才出现的结果。

## 4. 三因子学习：局部相关性还需要一个结果信号

### 4.1 从两因子到三因子

如果共同发放就一定增强，网络记录的只是相关性，而不是行为价值。动物还需要知道这次活动的结果：奖励、惩罚、新奇性、内部稳态变化或预测误差都可能改变突触可塑性。

因此，三因子规则在突触前、突触后活动之外，引入一个调制因子：

$$
\Delta w_{ij}(t)=\eta M(t)e_{ij}(t).
$$

这里的 $e_{ij}(t)$ 是由局部活动形成的资格迹，$M(t)$ 是奖励、预测误差或神经调质信号。这个公式不是说所有神经调质都以一个标量乘法实现，而是把三类信息的功能关系写清楚：

- 局部活动提出“哪些突触可能参与了这次计算”；
- 资格迹暂时保存这段历史；
- 调制信号决定这些候选改变是否值得写入长期权重。

![三因子学习规则：局部的突触前后活动产生资格迹，神经调质信号决定权重更新](/assets/img/posts/neuro-learning-three-factor.svg)

*图 2。三因子规则把局部活动形成的资格迹与调制信号结合。图中的乘法是算法层面的表达，生物过程可能由多步细胞机制共同实现。本文自绘。*

### 4.2 奖励调制 STDP

Izhikevich 提出的模型把毫秒尺度的 STDP 与慢得多的多巴胺信号连接起来：突触先保留局部活动留下的资格，延迟奖励到来时再选择性地强化这些突触，从而缓解 distal reward problem。[论文页面](https://izhikevich.org/publications/dastdp.htm)

奖励调制 STDP 的一个简化写法是：

$$
\begin{aligned}
\tau_e\dot e_{ij}(t)&=-e_{ij}(t)+F\big(s_i(t),s_j(t)\big),\\
\Delta w_{ij}(t)&=\eta\,\delta(t)e_{ij}(t),
\end{aligned}
$$

其中 $F$ 可以是 STDP 窗口，$\delta(t)$ 可以是奖励预测误差。这个写法比“奖励直接乘在 STDP 上”更准确，因为它明确表示局部活动需要先被保存。

### 4.3 神经调质不只是正负奖励开关

多巴胺常被用来表示奖励预测误差，但乙酰胆碱、去甲肾上腺素和其他调制信号还可能改变神经元兴奋性、突触可塑性的阈值、探索程度和学习时间窗。因此，更合理的理解是：神经调质改变“什么样的局部活动有资格被写入”，而不只是直接指定“权重增加”或“权重减少”。

三因子规则的理论综述可参考 [Frémaux 与 Gerstner（2015）](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full)。

## 5. 资格迹：把延迟结果和过去的活动联系起来

假设动物先看到气味，几秒后才获得糖水。糖水出现时，真正需要改变的可能是几秒前参与气味表征的突触。若突触只能查看奖励到来时的脉冲，它已经无法知道哪些局部活动与奖励有关。

资格迹为每个突触增加一个短期记忆变量：

$$
\tau_e\frac{de_{ij}}{dt}=-e_{ij}+f(x_i,y_j).
$$

活动发生时，$e_{ij}$ 上升；随后它逐渐衰减。调制信号到来时，更新量由当时仍然存在的资格迹决定：

$$
\Delta w_{ij}(t)=\eta M(t)e_{ij}(t).
$$

![资格迹在气味出现后逐渐衰减，延迟到来的奖励只有在迹线尚未消失时才能调制相关突触](/assets/img/posts/neuro-learning-eligibility.svg)

*图 3。气味诱发的局部活动先留下资格迹；延迟奖励到来时，仍有资格迹的突触才可能被更新。时间长度只是概念示意。本文自绘。*

资格迹可以是抽象算法变量，也可以对应钙浓度、受体状态或其他突触生化过程。它解决的是“过去哪些突触最近活跃过”，并不自动解决“整个网络应该如何分配误差”。资格迹的时间常数、调制信号的空间范围以及突触之间的竞争仍需要由模型和实验共同约束。

在强化学习语言中，第三因子可以写成奖励预测误差或时序差分误差。局部突触活动形成资格迹，误差信号则判断这段活动带来的结果高于还是低于预期。

果蝇蘑菇体中的 Kenyon 细胞—多巴胺神经元—输出神经元回路，是研究这种机制的一个重要系统。相关模型把气味表征、奖励预测误差和突触变化放在同一个回路中讨论，例如 [关于果蝇蘑菇体强化预测误差的模型研究](https://www.nature.com/articles/s41467-021-22592-4)。这些模型有助于提出可检验的机制，但不能直接等同于已经被实验完全证实的突触更新公式。

## 6. e-prop 和 SuperSpike：局部学习规则如何训练循环 SNN

### 6.1 e-prop：资格迹乘以学习信号

循环脉冲网络的 BPTT 需要沿时间反向传播误差，并保存较长时间内的网络状态。Bellec 等人提出的 e-prop 试图把这个计算拆成两部分：

1. 每个突触根据局部前、后活动在线计算资格迹；
2. 一个较低维的学习信号在需要时与资格迹相乘，形成权重更新。

抽象写法为：

$$
\Delta w_{ij}(t)\approx L_j(t)e_{ij}(t),
$$

其中 $e_{ij}(t)$ 只依赖突触附近的活动和神经元动力学，$L_j(t)$ 是到达突触后神经元或相关回路的学习信号。与 BPTT 的精确梯度相比，e-prop 使用可在线计算的近似分解。

![e-prop 将突触局部计算的资格迹与到达神经元的学习信号结合，形成在线参数更新](/assets/img/posts/neuro-learning-eprop.svg)

*图 4。e-prop 的计算分工：资格迹由局部神经动力学在线得到，学习信号由输出误差或奖励相关回路提供。本文根据 [Bellec 等人（2020）](https://www.nature.com/articles/s41467-020-17236-y) 的方法自绘。*

### 6.2 e-prop 和三因子规则是什么关系

从功能分解看，e-prop 可以放进三因子框架：突触前活动、突触后活动和神经元级学习信号共同决定更新；资格迹则保存局部时间信息。但两者不能简单画等号：

- 三因子规则是一类关于信息来源的宽泛框架；
- e-prop 是针对循环脉冲网络的在线梯度近似方法；
- 奖励型 e-prop 使用奖励信号，而标准 e-prop 也可以使用监督误差或任务学习信号。

因此，e-prop 是三因子思想的一种算法化实现，但不是三因子学习的唯一形式。

### 6.3 SuperSpike：用替代梯度构成三因子更新

SuperSpike 使用平滑的替代梯度近似脉冲函数的不可导部分，并将突触前活动、突触后误差信号和局部资格迹组织成在线更新规则。[Zenke 与 Ganguli（2018）](https://zenkelab.org/wp-content/uploads/2018/03/ZenkeGanguli2018_SuperSpike.pdf)还提供了相应的实现代码，适合用来说明“局部更新规则”与“训练深层 SNN”之间的工程桥梁：[代码仓库](https://github.com/fzenke/pub2018superspike)。

SuperSpike 和 e-prop 的共同点是都保留了局部资格迹，差别在于学习信号、反馈形式和梯度近似方式。它们说明，神经启发算法不一定要在“纯 STDP”和“完整 BPTT”之间二选一，而可以设计中间层次的在线学习规则。

## 7. 预测性可塑性和树突计算：局部误差从哪里来

### 7.1 树突不是一个简单的加法器

人工神经元通常写成：

$$
y=\phi\left(\sum_iw_ix_i+b\right).
$$

真实神经元的树突具有空间结构。不同分支可以接收前馈、反馈和侧向输入，局部非线性还可能在信号到达胞体前完成筛选。因此，一个神经元内部可能同时存在多个时间尺度和多个局部计算单元。

这给局部学习提供了一个候选机制：如果反馈或目标信号到达特定树突分支，神经元就可能在本地比较“前馈预测”和“反馈状态”，再把差异转换为突触变化。

### 7.2 树突预测胞体发放

Urbanczik 和 Senn 提出的模型让树突电位预测胞体是否会发放，树突预测与实际胞体脉冲之间的不一致驱动突触可塑性。[论文](https://pubmed.ncbi.nlm.nih.gov/24507189/)

这个思路把“局部误差”解释成树突对胞体活动的预测误差：突触不需要读取完整的网络损失，而是根据所在分支中的局部预测和实际结果调整自己。

### 7.3 分离树突与局部误差

Guerguiev、Lillicrap 与 Richards 的模型把基底树突看作前馈输入区，把顶端树突看作反馈或目标输入区，并让局部电位差参与学习。[论文](https://elifesciences.org/articles/22901)

![分离树突模型中，基底树突接收前馈输入，顶端树突接收反馈信号，两者在神经元内形成局部学习信息](/assets/img/posts/neuro-learning-dendrite.svg)

*图 5。分离树突模型中的信号分工，不代表所有生物神经元都按此方式计算误差。本文根据 [Guerguiev 等人（2017）](https://elifesciences.org/articles/22901) 的模型自绘。*

需要保持谨慎：树突参与计算是实验事实，但某个具体的局部误差学习算法是否在脑中实现，还取决于反馈连接、离子通道、抑制性回路和分子可塑性机制。树突模型应被看作一组候选机制，而不是统一的生物学习定律。

### 7.4 预测性塑性：从相关性转向预测误差

Nature Communications 2023 的一项研究让单个神经元预测未来输入脉冲，并根据预测误差调整突触，展示了预测性学习如何产生序列提前反应和时间信用分配。[论文](https://www.nature.com/articles/s41467-023-40651-w)

另一个值得加入的例子是 Nature Neuroscience 2023 的工作，它把 Hebbian plasticity 与 predictive plasticity 结合到深层感觉网络中，并在脉冲模型中加入局部电压相关的可塑性项和抑制性 STDP。[论文](https://www.nature.com/articles/s41593-023-01460-y)

这类方法的共同方向是：突触不只记录“前后神经元是否一起发放”，还尝试记录“当前活动是否偏离了一个局部预测”。这使 Hebbian 学习与预测编码、误差驱动学习产生了连接。

### 7.5 树突动力学不等于树突学习

Zheng 等人的工作把不同时间常数的树突分支加入 SNN，用于处理多时间尺度动态。[论文](https://www.nature.com/articles/s41467-023-44614-z)

它主要说明**树突动力学结构如何帮助表示时间信息**。论文中的参数通过 BPTT 训练，因此不能单独作为局部突触更新的例子。阅读这类论文时，应该分别记录：

- 树突结构是否提供了新的神经动力学；
- 参数更新是否依赖全局反向传播；
- 运行过程中突触是否继续根据局部活动改变。

## 8. 持续学习：新知识如何不覆盖旧知识

动物不会完成一个任务后清空大脑，再开始下一个任务。新的经验会改变已有回路，但重要的旧知识又不能被完全覆盖。这就是持续学习中的灾难性遗忘问题。

### 8.1 EWC：重要参数需要更高的改变代价

弹性权重固化（elastic weight consolidation，EWC）为旧任务的重要参数增加惩罚：

$$
\mathcal L(\theta)=\mathcal L_B(\theta)+\frac{\lambda}{2}\sum_iF_i(\theta_i-\theta_{A,i}^{*})^2.
$$

其中 $\theta_A^*$ 是旧任务参数，$F_i$ 表示参数重要性。它把“重要突触更稳定、不重要突触更容易改变”的思想转化为可计算规则。[Kirkpatrick 等人的原始工作](https://doi.org/10.1073/pnas.1611835114)是持续学习文献中的经典起点。

### 8.2 Synaptic Intelligence：在学习过程中估计突触贡献

Synaptic Intelligence 不等到任务结束后才估计所有参数的重要性，而是让每个突触在线积累自己的参数变化与损失下降之间的关系，再把高贡献参数巩固起来。[论文](https://proceedings.mlr.press/v70/zenke17a.html)

它和生物启发的联系在于：重要性可以按突触分别记录，更新过程不需要保存完整的过去数据集。但它仍然是工程化的近似，不能直接等同于某种已被确认的突触分子机制。

### 8.3 Differentiable Plasticity：运行时局部，训练时全局

Miconi、Stanley 和 Clune 的 Differentiable Plasticity 让网络在运行过程中使用 Hebbian 可塑连接，同时用反向传播学习塑性系数。[论文](https://proceedings.mlr.press/v80/miconi18a.html)

它很适合用来澄清一个常被混淆的区别：

- **运行时局部性**：当前输入到来后，突触可以根据局部活动更新；
- **元参数训练方式**：塑性系数可以提前用反向传播离线优化。

所以，“网络运行时使用局部可塑性”并不意味着整个模型从训练到部署都完全不依赖反向传播。

### 8.4 Loss of plasticity：网络也会失去学习能力

Dohare 等人在 Nature 2024 的工作中讨论了深度持续学习中的 loss of plasticity：网络在连续任务中不仅会遗忘过去，还可能逐渐失去适应新任务的能力。[论文](https://www.nature.com/articles/s41586-024-07711-7)

这一结果把问题从“怎样保护旧知识”推进到“怎样保持新知识仍然能够写入”。它可以与突触巩固、元可塑性、神经发生或少量单元重置联系起来。

### 8.5 神经调质门控和 PFC–MD 模型

PFC–MD 类模型使用与丘脑调节相关的门控机制，在不同任务之间保护当前重要的前额叶活动，并减少相互干扰。[模型论文](https://www.nature.com/articles/s41467-024-52289-3)

这类工作适合作为连接“神经调质”和“持续学习”的案例：调制信号不一定直接改变所有突触，而可能先改变哪些神经元、哪些通路和哪些记忆状态能够进入当前计算。

![持续学习中，重要的旧任务参数受到巩固约束，其他参数仍可适应新任务](/assets/img/posts/neuro-learning-consolidation.svg)

*图 6。突触巩固的概念示意：旧任务的重要参数更难被新任务覆盖；EWC 用参数重要性加权的惩罚项实现这一思想。本文自绘。*

## 9. 果蝇全脑模型和数字生命：从“可运行”走向“可学习”

果蝇模型为神经启发学习提供了一个清楚的参照系。连接组提供结构约束，LIF 或其他动力学模型把结构变成可运行的回路，行为任务则检验回路是否产生合理的感觉—运动输出。但“模型可以运行”和“模型可以通过经验学习”是两件不同的事。

基于成年果蝇全脑连接组的模型可以把神经元连接、兴奋/抑制作用和神经动力学放在同一个网络中，用来研究刺激如何经过全脑回路产生行为输出。[Shiu 等人的 Nature 论文](https://www.nature.com/articles/s41586-024-07763-9)是这一方向的重要案例。若突触权重在模拟开始前就固定，模型主要复现的是连接组约束下的动力学，而不是完整的学习型数字生命。

视觉系统的连接组约束模型也说明了类似问题。Lappalainen 等人的工作把果蝇视觉回路的连接结构与视觉运动任务结合起来，预测跨视觉系统的神经活动。[论文](https://www.nature.com/articles/s41586-024-07939-3)

这类模型与局部突触可塑性之间还隔着一层：模型在训练阶段怎样得到参数，与模型在运行阶段能否根据经验修改参数，并不是同一个问题。要把果蝇数字脑推进到“可学习”的层次，至少需要补上四个环节：

1. **可修改的突触**：明确哪些连接具有可塑性，更新变量是什么；
2. **局部活动记录**：让突触根据前、后神经元活动生成资格迹；
3. **调制或误差信号**：由多巴胺样信号、奖励预测误差或内部稳态信号筛选更新；
4. **身体—环境闭环**：让动作改变环境，环境反馈再影响下一轮神经活动和学习。

这四点把“连接组的结构复现”与“学习机制的复现”连接起来。果蝇蘑菇体等小型回路尤其适合作为中间尺度：它们比全脑模型更容易定位神经元、行为和奖励信号之间的关系，又比单个突触模型更接近具身行为。

![果蝇数字生命从固定连接组和神经动力学，加入可塑突触与身体环境反馈，形成闭环学习](/assets/img/posts/neuro-learning-digital-fly.svg)

*图 7。从可运行的果蝇回路走向可学习的数字生命，需要把可塑性、调制信号和身体—环境反馈纳入同一闭环。本文自绘。*

## 10. 几类学习机制放在一起比较

| 机制 | 主要信息来源 | 时间信用分配 | 更新的局部性 | 主要优点 | 主要限制 |
| --- | --- | --- | --- | --- | --- |
| Hebbian learning | 突触前、后活动 | 很短 | 高 | 简单、可在线更新 | 缺少任务目标，容易失控 |
| STDP | 前、后脉冲的时间差 | 典型为毫秒级 | 高 | 能利用脉冲顺序 | 难处理延迟奖励和全局目标 |
| 奖励调制 STDP | STDP 资格迹 + 奖励误差 | 由资格迹决定 | 局部迹线 + 调制 | 能把相关性连接到行为结果 | 需要解释奖励信号的来源和范围 |
| 三因子规则 | 局部活动 + 调制信号 | 由资格迹决定 | 通常较高 | 统一 Hebbian、STDP 和奖励学习 | 是框架，不是唯一算法 |
| SuperSpike | 局部活动 + 替代梯度 + 反馈 | 可覆盖时间序列 | 局部迹线 + 反馈 | 可以训练多层 SNN | 反馈和替代梯度仍是工程近似 |
| e-prop | 资格迹 + 学习信号 | 可覆盖长时间序列 | 局部迹线 + 学习信号 | 在线训练循环 SNN | 学习信号仍需实现或近似 |
| 树突局部误差 | 前馈、反馈和分支电位 | 取决于树突动力学 | 倾向局部 | 可能缓解全局反向传播问题 | 生物实现尚未统一 |
| EWC/SI/突触巩固 | 参数重要性和旧任务记忆 | 长期 | 可以按参数分解 | 抑制灾难性遗忘 | 不等于完整的生物突触模型 |

这里最需要避免的误解是：局部性不是一个二元标签。一个规则可以让突触更新只依赖局部资格迹，但同时仍需要一个来自输出层的学习信号；一个规则也可以在运行时使用局部可塑性，却在训练塑性系数时依赖反向传播。

## 11. 我的理解：神经启发学习是一条信用分配链

把这些论文放在一起看，神经启发学习不是寻找一个名字最像大脑的单一算法，而是在回答信用分配链上的不同问题：

- **局部活动发生了什么？** Hebbian learning 和 STDP 记录突触两端的相关性与时间顺序；
- **这段活动是否值得保留？** 三因子规则引入奖励、预测误差或其他神经调质；
- **奖励来晚了怎么办？** 资格迹把过去的局部活动暂时保存下来；
- **局部误差信号从哪里来？** e-prop、SuperSpike 和树突模型尝试用资格迹、反馈或分支电位近似完成信用分配；
- **新知识如何不覆盖旧知识？** 突触巩固、元可塑性和多时间尺度记忆限制参数漂移；
- **这些规则能否产生行为？** 数字生命模型还需要把可塑突触放入身体—环境闭环。

因此，我暂时把这套机制概括为：**局部活动提出候选改变，资格迹保存因果历史，调制信号筛选改变，树突和反馈提供局部目标，巩固机制决定改变保留多久，身体反馈检验这些改变是否真的有用。**

这也解释了为什么“脉冲神经元”本身不足以代表神经启发学习：脉冲改变了网络的动力学表示，只有当权重更新、时间信用分配和持续学习也被重新设计时，网络才真正触及神经启发学习机制。

## 12. 论文阅读路线：按机制而不是按期刊阅读

如果要把这篇文章继续扩展成论文阅读系列，我建议优先细读下面八篇：

1. **Diehl & Cook（2015）**：先搭一个真正使用 STDP、侧向抑制和自适应阈值的 SNN；
2. **Izhikevich（2007）**：理解资格迹怎样把 STDP 与延迟奖励连接起来；
3. **Urbanczik & Senn（2014）**：理解树突预测如何形成局部误差；
4. **Zenke et al.（2017）**：理解突触如何在线记录任务重要性；
5. **Miconi et al.（2018）**：理解运行时局部可塑性与离线元学习的结合；
6. **Zenke & Ganguli（2018）**：理解替代梯度和三因子在线更新；
7. **Bellec et al.（2020）**：理解 e-prop 如何处理循环网络中的时间信用分配；
8. **Nature Neuroscience（2023）和 Dohare et al.（2024）**：分别补足预测性塑性与持续学习中的可塑性衰退。

每篇论文可以固定记录五个问题：

1. 它借鉴了哪一种神经机制？
2. 突触更新所需的信息在哪里产生？
3. 更新是否在运行时局部完成？
4. 模型是否在行为过程中继续学习，还是只在训练阶段优化参数？
5. 实验结果证明了什么，哪些部分仍然只是生物启发的工程假设？

这样的记录方式可以避免把神经动力学、网络连接结构和参数学习规则混在一起，也能看清 Nature 论文、会议论文和经典模型在整条机制链上的不同作用。

## 参考文献与延伸阅读

- [Diehl & Cook（2015），Unsupervised learning of digit recognition using STDP](https://pmc.ncbi.nlm.nih.gov/articles/PMC4522567/)
- [Izhikevich（2007），Solving the Distal Reward Problem through Linkage of STDP and Dopamine Signaling](https://izhikevich.org/publications/dastdp.htm)
- [Urbanczik & Senn（2014），Learning by the Dendritic Prediction of Somatic Spiking](https://pubmed.ncbi.nlm.nih.gov/24507189/)
- [Frémaux & Gerstner（2015），Neuromodulated STDP and Three-Factor Learning Rules](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full)
- [Guerguiev, Lillicrap & Richards（2017），Towards deep learning with segregated dendrites](https://elifesciences.org/articles/22901)
- [Zenke, Poole & Ganguli（2017），Continual Learning Through Synaptic Intelligence](https://proceedings.mlr.press/v70/zenke17a.html)
- [Miconi, Stanley & Clune（2018），Differentiable plasticity](https://proceedings.mlr.press/v80/miconi18a.html)
- [Zenke & Ganguli（2018），SuperSpike](https://zenkelab.org/wp-content/uploads/2018/03/ZenkeGanguli2018_SuperSpike.pdf)
- [Bellec et al.（2020），A solution to the learning dilemma for recurrent networks of spiking neurons](https://www.nature.com/articles/s41467-020-17236-y)
- [Sequence anticipation and spike-timing-dependent plasticity emerge from a predictive learning rule（2023）](https://www.nature.com/articles/s41467-023-40651-w)
- [The combination of Hebbian and predictive plasticity learns invariant object representations（2023）](https://www.nature.com/articles/s41593-023-01460-y)
- [Zheng et al.（2024），Temporal dendritic heterogeneity incorporated with spiking neural networks](https://www.nature.com/articles/s41467-023-44614-z)
- [Dohare et al.（2024），Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)
- [Shiu et al.（2024），A Drosophila computational brain model reveals sensorimotor processing](https://www.nature.com/articles/s41586-024-07763-9)
- [Lappalainen et al.（2024），Connectome-constrained networks predict neural activity across the fly visual system](https://www.nature.com/articles/s41586-024-07939-3)
- [果蝇蘑菇体中的强化预测误差模型](https://www.nature.com/articles/s41467-021-22592-4)
