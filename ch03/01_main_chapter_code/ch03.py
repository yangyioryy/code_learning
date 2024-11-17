"""
编码注意力机制

对attention机制来说，解码器会在产生输出时某个token时会有区别地关注输入的所有token
对Transform框架中的self-attention来说，每一个输入的token会和其他token交互，确定它们的相关性，从而对输入增强（即关注了上下文的信息）

"""

from importlib.metadata import version                                 #导入依赖库
print("torch version:", version("torch"))                              #检测


"""
1.用自注意力机制处理input embedding(不带可训练的权重)

首先用要处理的embedding作为query，与其他embedding向量点乘，计算注意力得分，
然后对其进行归一化，得到总计为 1 的注意力权重，
最后与对应input embedding相乘得到上下文向量（context vector）
"""

###1.1 无训练权重的简单自注意力机制
##     实验是对第二个input embedding进行了自注意力增强处理

import torch

inputs = torch.tensor(                                                 #input embeddings，假设是输入经过了一个三维的嵌入
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]                                                      #将第二个embedding作为query向量
attn_scores_2 = torch.empty(inputs.shape[0])                           #inputs.shape[0]就是返回列表长度，并初始化一个空的列表
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)                           #两个一维向量做点积
print(attn_scores_2)                                                   #返回各位置embedding对2的注意力分数（tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])）

#验证dot语法的用法
res = 0.                                                               #初始化一个浮点数
for idx, element in enumerate(inputs[0]):                              #遍历第一个embedding
    res += inputs[0][idx] * query[idx]                                 #将第一个embedding向量与对应query位置相乘
print(res)                                                             #tensor(0.9544)
print(torch.dot(inputs[0], query))                                     #tensor(0.9544)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()               #将注意力分数标准化得到注意力矩阵
print("Attention weights:", attn_weights_2_tmp)                        #（Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])）
print("Sum:", attn_weights_2_tmp.sum())                                #（Sum: tensor(1.0000)）

def softmax_naive(x):                                                  #使用softmax函数进行归一化处理（更善于处理极值，并且在训练过程中具有更理想的梯度特性）
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)                    #将注意力分数列表输入softmax函数，得到注意力权重列表
print("Attention weights:", attn_weights_2_naive)                      #（Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])）
print("Sum:", attn_weights_2_naive.sum())                              #（Sum: tensor(1.)）

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)                   #使用pytorch封装的softmax函数处理更好
print("Attention weights:", attn_weights_2)                            #同上
print("Sum:", attn_weights_2.sum())                                    #同上

query = inputs[1]                                                      #以第二个embedding作为query向量
context_vec_2 = torch.zeros(query.shape)                               #初始化一个0矩阵
for i,x_i in enumerate(inputs):                            
    context_vec_2 += attn_weights_2[i]*x_i                             #将各embedding与对应权重相乘，得到注意力增强后的输入
print(context_vec_2)                                                   #（tensor([0.4419, 0.6515, 0.5683])）


###1.2 对每一个input embedding进行自注意力增强
attn_scores = torch.empty(6, 6)                                        #初始化一个6*6的空矩阵
for i, x_i in enumerate(inputs):                                       #query向量
    for j, x_j in enumerate(inputs):                                   #input向量
        attn_scores[i, j] = torch.dot(x_i, x_j)                        #计算每个embedding的注意力分数
print(attn_scores)                                                     #（6*6的注意力分数矩阵）

attn_scores = inputs @ inputs.T                                        #上述代码直接用一个矩阵乘法就可
print(attn_scores)                                                     #（同上）

attn_weights = torch.softmax(attn_scores, dim=-1)                      #在一维计算注意力权重
print(attn_weights)                                                    #（标准化后结果）

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])      #检验计算结果是否为1
print("Row 2 sum:", row_2_sum)                                         #（Row 2 sum: 1.0）
print("All row sums:", attn_weights.sum(dim=-1))                       #dim=-1就是对最后一维向量处理，在这里就是1（All row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])）
  
all_context_vecs = attn_weights @ inputs                               #直接用矩阵乘法得到上下文向量
print(all_context_vecs)                                                #（打印全部的上下文向量）
print("Previous 2nd context vector:", context_vec_2)                   #之前计算的第二个embedding的上下文向量（Previous 2nd context vector: tensor([0.4419, 0.6515, 0.5683])）



"""
2.用自注意力机制处理input embedding(有可训练的权重)

Q、K、V三个矩阵
Q@K得到注意力分数      归一化得注意力权重
context向量 = 注意力权重@V
"""

### 2.1 逐步计算注意力权重

x_2 = inputs[1]                                                        #第二个embedding
d_in = inputs.shape[1]                                                 #input中每个embedding的大小（3）
d_out = 2                                                              #输出维度的大小（2），之前的out维度就是1

torch.manual_seed(123)                                                 #设置随机数种子
#初始化q，k，v权重矩阵
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)#设置一个3*2维的张量作为可学习的参数
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)#设置一个3*2维的张量作为可学习的参数
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)#设置一个3*2维的张量作为可学习的参数

query_2 = x_2 @ W_query                                                # 关于第二个embedding的query向量
key_2 = x_2 @ W_key                                                    # 关于第二个embedding的key向量
value_2 = x_2 @ W_value                                                # 关于第二个embedding的value向量
print(query_2)                                                         # 将一个原始的三维向量转化为二维的上下文向量（tensor([0.4306, 1.4551])）

keys = inputs @ W_key                                                  #同样获取整个input向量的key向量矩阵
values = inputs @ W_value                                              #获取整个input向量的value向量矩阵
print("keys.shape:", keys.shape)                                       #(keys.shape: torch.Size([6, 2]))
print("values.shape:", values.shape)                                   #(values.shape: torch.Size([6, 2]))

keys_2 = keys[1]              
attn_score_22 = query_2.dot(keys_2)                                    #第二个向量的注意力分数
print(attn_score_22)                                                   #（tensor(1.8524)）

attn_scores_2 = query_2 @ keys.T                                       #第二个元素的注意力分数矩阵
print(attn_scores_2)                                                   #（tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])）

d_k = keys.shape[1]                                                    #用注意力分数的维度的平方根对注意力分数进行一个缩放
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)       #在最后一个维度求注意力权重
print(attn_weights_2)                                                  #（tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])）

context_vec_2 = attn_weights_2 @ values                                #将的得到的注意力权重与对应values相乘
print(context_vec_2)                                                   #(tensor([0.3061, 0.8210]))



### 2.2 一个完整的自注意力层
import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()                                           #初始化
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))         #设置q矩阵
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))         #key矩阵
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))         #value矩阵

    def forward(self, x):
        keys = x @ self.W_key                                        #计算key向量
        queries = x @ self.W_query                                   #计算query向量
        values = x @ self.W_value                                    #计算value向量
        
        attn_scores = queries @ keys.T # omega                       #q*k得注意力分数 
        attn_weights = torch.softmax(                                #归一化的注意力权重
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values                          #得context向量
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))



class SelfAttention_v2(nn.Module):                                 #另一种自注意力的实现

    def __init__(self, d_in, d_out, qkv_bias=False):               #禁用bias则视为矩阵乘法
        super().__init__()              
        self.W_query = nn.Linear(d_in,d_out, bias=qkv_bias)        #用线性层Linear
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)                                       #向线性层传入一个值，就实现了上述的矩阵乘法功能
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T                            #注意力分数
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)#注意力权重

        context_vec = attn_weights @ values                       #上下文向量
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)                             #即使设置一样的随机数种子，不同类别的随机数还是不一样的
print(sa_v2(inputs))


"""
3.使用因果注意力（相对于之前得自注意力）来隐藏未来的词语

只考虑和自己前面的词的关系

将注意力权重矩阵上三角置0，或将注意力分数矩阵上三角置-∞
再使用dropout层（dropoout还可以防止过拟合）                  ->只用于训练过程，推理过程中不使用
"""


### 3.1 将注意力权重矩阵上三角置0，或将注意力分数矩阵上三角置-∞

queries = sa_v2.W_query(inputs)                                              #获取q
keys = sa_v2.W_key(inputs)                                                   #获取k
attn_scores = queries @ keys.T                                               #相乘的注意力分数
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)      #归一化的注意力矩阵
print(attn_weights)                                                          #（6*6的矩阵）
 
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))         #利用tril函数生成掩码（对角线和它的下面都是1，下面都是0）
print(mask_simple)                                                           #下三角矩阵

masked_simple = attn_weights*mask_simple                                     #直接相乘将原注意力权重矩阵对角线上面的变为0
print(masked_simple)
#这种方法让归一化后的矩阵每一行的和又不是1了

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums                                #每一行再除以这一行的总和，重归一
print(masked_simple_norm)                 

#另一种更有效的掩码策略，将注意力分数矩阵的上三角区域置为-∞
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)    #加了diagonal=-1后变成了一个上三角矩阵
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)                    #将原来mask矩阵为1的部分填-∞
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)           #再归一化得注意力权重矩阵
print(attn_weights)



### 3.2 使用dropout掩盖过度的注意力，防止过拟合

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)                                             #设置一个dropout层，比为0.5
example = torch.ones(6, 6)                                                  # 设置一个全一的矩阵
print(dropout(example))                                                     #随机将一半的置置为0，其他的值放大为原来的两别

torch.manual_seed(123)
print(dropout(attn_weights))                                                #对注意力权重进行dropout


### 3.3 实现一个完整的因果注意力层(支持批次输入)

batch = torch.stack((inputs, inputs), dim=0)                                #创建一个批次的数据（将两个input向量，在另一个维度进行堆叠）
print(batch.shape)                                                          #（torch.Size([2, 6, 3])）


class CausalAttention(nn.Module):                                          

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out                                                 #dropout率
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)               #q矩阵
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)               #k矩阵
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)               #v矩阵
        self.dropout = nn.Dropout(dropout)                                 #dropout层
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New
                                                                           #注册一个缓冲区，不作为模型参数，设置一个掩码矩阵
    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b              #获取一个批次的所有信息
        keys = self.W_key(x)                                               #key
        queries = self.W_query(x)                                          #query
        values = self.W_value(x)                                           #value

        attn_scores = queries @ keys.transpose(1, 2)                       #注意力分数 transpose(1, 2)将第一维和第二维交换
        attn_scores.masked_fill_(                                          #将注意力分数进行一个掩码
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
        attn_weights = torch.softmax(                                      #归一化的注意力权重
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)                          #通过dropout层

        context_vec = attn_weights @ values                                #得到context向量
        return context_vec    
 
torch.manual_seed(123)                                                     #设置随机数种子
context_length = batch.shape[1]                                            #批次中输入的长度
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)                                                   
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


"""
4.多头注意力层

直接堆叠单头注意力层
"""


### 4.1 堆叠单头注意力层

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(                                                             #设置一系列的因果注意力层
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)                             #输入会独立经过多个注意力层，然后在最后一个维度进行拼接    


torch.manual_seed(123)                                                                         #设置随机数种子
context_length = batch.shape[1]                                                                # This is the number of tokens 
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)                                               #相当于两个经过注意力层后的input堆叠而成


### 4.2 权重分割（重复上述的步骤，不再调用CausalAttention）

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"                                             #输出一定要可以被head数目拆分

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads                                                    # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)                                              # 将最后的答案通过这个线性层，进行整合
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)                                                                # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)                     #把d_out拆分为了多个注意力头
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)                                           #将最后得到的结果再进行一个线性的转化，并不是必须的

        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)