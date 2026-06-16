# 00. PyTorch Warmup | PyTorch 核心基础热身: 张量、前反向传播与 Embedding (Warmup)

***1.einops：可读的张量维度变换***
`einops` 是一个用于张量维度变换的库，可以用类似公式的字符串表达 `reshape / transpose / permute / repeat / reduce` 等操作。
<1>rearrange(x, 'b h s d -> b s (h d)')
b = batch
h = heads
s = sequence length
d = head dimension
(h d) = 把 heads 和 head_dim 合并

<2>reduce(x, 'b s d -> b d', 'mean')
对 `seq_len` 维度取平均，把 `[batch, seq_len, dim]` 变成 `[batch, dim]`

<3>repeat(x, 'b d -> b s d', s=10)
把 `[batch, dim]` 扩展成 `[batch, seq_len, dim]`

==Note: 这些变量名在不同特征情况下代表意思不同，不是固定的，比如h可表示heads也可表示height，只是一个表示当前维度的名字，方便理解。==

***2.permute & reshape: 改变维度***
`x_native=x.permute(0,2,3,1).reshape(`

        x.shape[0],

        x.shape[2] * x.shape[3],

        x.shape[1]

`)`

permute将维度[batch_size, channels, height, width] 改成 [batch_size,  height, width, channels], reshape将height和width乘积。

***3.embedding***
`Neural Network `理解不了输入的文本，只能通过`Tokenizer` 将输入的文本-->token ids, 再通过`Embedding`将ids转换成`Neural Network`可以理解的向量。

`Embedding 表`: [vocab_size, hidden_dim] 每一行对应一个token向量。Embedding要做的就是取出表中的某一行向量。

Embedding 在数学上等价于：

```
one_hot(token_id) @ W
```
one_hot(2):第二行是1，其余都是0     one_hot(2) @ w取出w表的第二行。

**Pytorch写法：**
```
embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=hidden_dim
)

input_ids = torch.tensor([[10, 42, 99]])

x = embedding(input_ids)
```
==Note: `embedding.weight[input_ids]`手动根据index查表也可以达到同样的效果==

***4.Forward & Backward***
`Forward`: Linear + ReLu
`Backward`: Gradient Descent不断更新参数

$Z=X @ W^T + B$
$Y=ReLu(Z)$

$grad_x= grad_z @ w$
$grad_z=grad_z^T @ x$
$grad_b=grad_z .sum(dim=0)$每个batch分别求和就是对应的bias梯度

==Note: ctx 是Forward和Backward之间用来保存上下文的工具。==
==ctx.save_for_backward(x, weight, mask)==
==x, weight, mask = ctx.saved_tensors==


# 01. RMSNorm Tutorial | 均方根层归一化 (RMSNorm)

***1.RMSNorm***
去掉Mean的操作，直接用RMS归一化。
   $$ \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon} $$
   $$ y = \frac{x}{\text{RMS}(x)} \odot \gamma $$

***2.精度溢出问题***
FP16的最大精度只有65504，因此在进行计算（mean, variance）时需要先转换为float32，结果再转换成原精度。