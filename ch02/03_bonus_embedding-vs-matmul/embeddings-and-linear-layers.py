"""
比较嵌入层和线性层之间
"""


import torch                                                                #导入依赖库
print("PyTorch version:", torch.__version__)                                #检测

"""
1.使用嵌入层

直观上看，建立了id号和高维张量的一一对应关系，类似于数组
"""

idx = torch.tensor([2, 3, 1])                                              #假设input中有三个元素，tokenlize之后的情况是这样
num_idx = max(idx)+1                                                       #语料库元素个数=idx最大值+1
# The desired embedding dimension is a hyperparameter
out_dim = 5                                                                #嵌入层输出为一个五维的张量
torch.manual_seed(123)                                                     #设置随机种子
embedding = torch.nn.Embedding(num_idx, out_dim)                           #指定语料库元素个数和输出维度
embedding.weight                                                           #输出一个4*5的张量
embedding(torch.tensor([1]))                                               #标号1张量（tensor([[ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]],grad_fn=<EmbeddingBackward0>)）
embedding(torch.tensor([2]))                                               #标号2张量（tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315]],grad_fn=<EmbeddingBackward0>）
idx = torch.tensor([2, 3, 1])
embedding(idx)                                                             #将原始的文本id通过嵌入层(3*5的张量)


"""
2.使用线性层

先将文本id转化的独热向量编码，再通过线性层转化为所需维度
"""

onehot = torch.nn.functional.one_hot(idx)                                  #将原始文本id转化的独热向量
onehot                                                                     #tensor([[0, 0, 1, 0], [0, 0, 0, 1],[0, 1, 0, 0]])
torch.manual_seed(123)                                                     #设置随机数种子
linear = torch.nn.Linear(num_idx, out_dim, bias=False)                     #通过一个线性层，指定输维度和输出维度
linear.weight                                                              #原始线性层权重（一个5*4的矩阵W）
linear.weight = torch.nn.Parameter(embedding.weight.T)                     #重新赋值为和上述嵌入层相同的权重矩阵的转置（因为经过线性层时，实际上是和权重矩阵的转置相乘）

linear(onehot.float())                                                     #将独热向量通过这个线性层
embedding(idx)                                                             #将独热向量通过这个嵌入层



#二者结果是完全一样的，实际上也可以说明二者的本质是相同的