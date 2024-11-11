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
value_2 = x_2 @ W_value                                                #关于第二个embedding的value向量
print(query_2)                                                         #（tensor([0.4306, 1.4551])）