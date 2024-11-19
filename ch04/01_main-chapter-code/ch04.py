"""
从头开始实现一个GPT模型来生成文本
"""


#导入必要依赖库并检验
from importlib.metadata import version
import matplotlib
import tiktoken
import torch
print("matplotlib version:", version("matplotlib"))
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))
import torch
import torch.nn as nn

"""
1.大模型架构
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,                                       # 词汇表大小
    "context_length": 1024,                                    # 文本长度
    "emb_dim": 768,                                            # 嵌入层维度
    "n_heads": 12,                                             # 注意力的头数
    "n_layers": 12,                                            # 层数
    "drop_rate": 0.1,                                          # Dropout率
    "qkv_bias": False                                          # Query-Key-Value 偏移
}


import torch
import torch.nn as nn


# 定义一个简单的 DummyGPT 模型类，继承自 PyTorch 的 nn.Module。
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()                                                               # 调用父类的初始化方法。     
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])                   # 定义一个词嵌入层，输入是词表大小，输出是嵌入维度。
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])               # 定义一个位置嵌入层，输入是上下文长度，输出是嵌入维度。
        self.drop_emb = nn.Dropout(cfg["drop_rate"])                                     # 定义一个 dropout 层，用于防止过拟合。
        
        self.trf_blocks = nn.Sequential(                                                 # 使用一个占位的 TransformerBlock（后续定义），包含多个 Transformer 层。
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )                                                                                
        
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])                                 # 使用一个占位的 LayerNorm（后续定义）用于最终的规范化处理。
        self.out_head = nn.Linear(                                                       # 定义输出层，将嵌入维度映射回词表大小，bias 置为 False。
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):                                                           # 定义前向传播过程。
        batch_size, seq_len = in_idx.shape                                               # 获取输入的批量大小和序列长度。
        tok_embeds = self.tok_emb(in_idx)                                                # 通过词嵌入层将输入索引转换为对应的嵌入表示。
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))           # 通过位置嵌入层为每个位置添加位置信息。
        x = tok_embeds + pos_embeds                                                      # 词嵌入和位置嵌入相加。
        x = self.drop_emb(x)                                                             # 应用 dropout。
        x = self.trf_blocks(x)                                                           # 通过多层 TransformerBlock 处理。
        x = self.final_norm(x)                                                           # 通过最终的 LayerNorm 处（获取一个标准化的输出）。
        logits = self.out_head(x)                                                        # 应用输出层，将嵌入映射到词表大小的 logits。
        return logits


 
class DummyTransformerBlock(nn.Module):                                                  # 定义一个 DummyTransformerBlock 类，作为占位符,未定义实际功能 
    def __init__(self, cfg):
        super().__init__()  

    def forward(self, x):
        return x

 
class DummyLayerNorm(nn.Module):                                                        # 定义一个 DummyLayerNorm 类，作为占位符。未定义实际功能
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()  #   

    def forward(self, x):
        return x



import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")                                               #导入一个分词器，处理词汇表中没有的单词
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"
 
batch.append(torch.tensor(tokenizer.encode(txt1)))                                      #用分词器对两段文本进行编码
batch.append(torch.tensor(tokenizer.encode(txt2))) 
batch = torch.stack(batch, dim=0)                                                       #在第一个维度铺平
print(batch)                                                                            #（tensor([[6109, 3626, 6100,  345],
                                                                                        # [6109, 1110, 6622,  257]])）

torch.manual_seed(123)                                                                  #设置一个随机数种子
model = DummyGPTModel(GPT_CONFIG_124M)                                                  #调用刚才的模型
logits = model(batch)                                                                   #输入批次数据
print("Output shape:", logits.shape)                                                    #（torch.Size([2, 4, 50257])）
print(logits)    


"""
2.层标准化实现标准化激活（LayerNorm）操作， 均值->0 方差->1

概率论中的标准化函数
"""
  
torch.manual_seed(123)                                                                 # 设置随机数种子
batch_example = torch.randn(2, 5)                                                      # create 2 training examples with 5 dimensions (features) each     
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())                                      # 定义一个神经网络层，一个线性层加一个激活层                 
out = layer(batch_example)                                                             # 通过这个神经网络层（输入预处理后的结果）
print(out)

 #获取现在的平均和方差，不为0，1
mean = out.mean(dim=-1, keepdim=True)                                                 
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
# Mean:
#  tensor([[0.1324],
#         [0.2170]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[0.0231],
#         [0.0398]], grad_fn=<VarBackward0>)


out_norm = (out - mean) / torch.sqrt(var)                                            # 标准化函数（X-μ）/σ
print("Normalized layer outputs:\n", out_norm)
mean = out_norm.mean(dim=-1, keepdim=True)                                           #再次获取平均值和方差后就是0，1了
var = out_norm.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

torch.set_printoptions(sci_mode=False)                                               #禁用科学计数法
print("Mean:\n", mean)                                                               #0
print("Variance:\n", var)                                                            #1


class LayerNorm(nn.Module):                                                          #实现一个层标准化的类
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5                                                              #一个很小的常数，防止计算中除以0
        self.scale = nn.Parameter(torch.ones(emb_dim))                               #可学习的缩放参数，初始化为全1向量
        self.shift = nn.Parameter(torch.zeros(emb_dim))                              #可学习的平移参数，初始化为全0

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                                          #原始均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)                            #原始方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)                             #标准化后的数值
        return self.scale * norm_x + self.shift                                      #对输入进行合适的缩放和平移
    
ln = LayerNorm(emb_dim=5)                                                            #嵌入维度设置为5
out_ln = ln(batch_example)                                                           #对批次数据进行标准化

mean = out_ln.mean(dim=-1, keepdim=True)                                            
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)                                                               #（0）
print("Variance:\n", var)                                                            #（1）