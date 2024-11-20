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

在数据预处理后会经历一串的Transform Block，
每个Transform Block有一个标准化的过程，
然后再经过一个线性层转化为和vocabulary set一样大小
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

"""
3.使用RELU(Rectified Linear Unit)激活实现一个前馈网络

还有GELU和SwiGLU
"""

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
           
    def forward(self, x):                                                           #GELU(x)≈0.5⋅x⋅(1+tanh[2π−−√⋅(x+0.044715⋅x3)])  
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
 
import matplotlib.pyplot as plt                                                     #导入绘图所需的库
gelu, relu = GELU(), nn.ReLU()                                                      #获取GELU函数和RELU函数
# Some sample data
x = torch.linspace(-3, 3, 100)                                                      #x范围
y_gelu, y_relu = gelu(x), relu(x)                                                   #两个函数对应的函数值
 
plt.figure(figsize=(8, 3))                                                          #图表大小
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):         #分成两个图绘制
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()                                                                          #分别得到两个函数对应的函数图像


class FeedForward(nn.Module):                                                       #前馈网路实现
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(                                                #将多个层按顺序组合在一起
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),                          #扩大特征向量
            GELU(),                                                                 #激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),                          #还原回原来的维度
        )

    def forward(self, x):
        return self.layers(x)                                            
    

print(GPT_CONFIG_124M["emb_dim"])                                                  #(768)
ffn = FeedForward(GPT_CONFIG_124M)                                                
x = torch.rand(2, 3, 768)                                                          #input shape: [batch_size, num_token, emb_size]
out = ffn(x)
print(out.shape)                                                                   #(torch.Size([2, 3, 768]))




"""
4.添加快捷连接，用于缓解梯度消失问题

为梯度在网络中流动创建了另一条更短的路径
"""


class ExampleDeepNeuralNetwork(nn.Module):                                        # 定义一个深度神经网络类，继承自 nn.Module
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()  
        self.use_shortcut = use_shortcut                                          # 保存是否使用捷径连接（shortcut）的标志位
        self.layers = nn.ModuleList([                                             # 定义一个模块列表，用于存储多个神经网络层
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),     # 第一层，全连接 + GELU激活函数
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),     # 第二层，全连接 + GELU激活函数
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),     # 第三层，全连接 + GELU激活函数
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),     # 第四层，全连接 + GELU激活函数
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())      # 第五层，全连接 + GELU激活函数
        ])

    def forward(self, x):                                                         # 定义前向传播方法
        for layer in self.layers:                                                 # 遍历每一层
            layer_output = layer(x)                                               # 计算当前层的输出
            if self.use_shortcut and x.shape == layer_output.shape:               # 如果启用捷径连接，且形状匹配
                x = x + layer_output                                              # 将当前层输出与输入相加（捷径连接）
            else:
                x = layer_output                                                  # 否则直接将当前层输出作为新的输入
        return x                                                                  # 返回最后一层的输出


def print_gradients(model, x):                                                    # 定义一个函数用于打印模型的梯度
    # Forward pass
    output = model(x)                                                             # 前向传播，计算模型的输出
    target = torch.tensor([[0.]])                                                 # 定义目标值，这里是一个标量 0

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()                                                           # 使用均方误差（MSE）作为损失函数
    loss = loss(output, target)                                                   # 计算损失值

    # Backward pass to calculate the gradients
    loss.backward()                                                               # 反向传播，计算梯度

    for name, param in model.named_parameters():                                  # 遍历模型中的所有参数
        if 'weight' in name:                                                      # 筛选出权重参数
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")  
            # 打印权重的梯度绝对值的平均值



layer_sizes = [3, 3, 3, 3, 3, 1]                                                  #刚才定义的各层的维度
sample_input = torch.tensor([[1., 0., -1.]])                                      #输入
torch.manual_seed(123)                                                            #随机数种子
model_without_shortcut = ExampleDeepNeuralNetwork(                                #不采用shortcut
    layer_sizes, use_shortcut=False
)
print_gradients(model_without_shortcut, sample_input)


torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(                                  #采用shortcut
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)

#可以观察到采用后梯度值明显增大了



"""
5.把注意力层和线性层结合在Transform Block中

经过这样一个Transform Block后的输入向量 可视为context vector的再一次加强版
"""

from previous_chapters import MultiHeadAttention                                # 从之前的章节中导入多头注意力模块


class TransformerBlock(nn.Module):                                              # 定义 Transformer 块类，继承自 nn.Module
    def __init__(self, cfg):
        super().__init__()                                                      # 调用父类的初始化方法   
        self.att = MultiHeadAttention(                                          # 初始化多头注意力模块
            d_in=cfg["emb_dim"],                                                # 输入嵌入维度
            d_out=cfg["emb_dim"],                                               # 输出嵌入维度，与输入相同
            context_length=cfg["context_length"],                               # 上下文长度
            num_heads=cfg["n_heads"],                                           # 注意力头的数量
            dropout=cfg["drop_rate"],                                           # 注意力模块中的 Dropout 概率
            qkv_bias=cfg["qkv_bias"]                                            # 是否在 QKV 计算中加入偏置
        )
        self.ff = FeedForward(cfg)                                              # 定义前馈网络
        self.norm1 = LayerNorm(cfg["emb_dim"])                                  # 第一层的 LayerNorm，用于注意力块
        self.norm2 = LayerNorm(cfg["emb_dim"])                                  # 第二层的 LayerNorm，用于前馈块
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])                       # 用于快捷连接的 Dropout

    def forward(self, x):                                                       # 定义前向传播逻辑
        # Shortcut connection for attention block
        shortcut = x                                                            # 保存输入数据，用于添加快捷连接
        x = self.norm1(x)                                                       # 通过第一层的 LayerNorm
        x = self.att(x)                                                         # 通过多头注意力模块，输出形状 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)                                               # 应用 Dropout
        x = x + shortcut                                                        # 将原始输入与当前输出相加，形成快捷连接

        # Shortcut connection for feed forward block
        shortcut = x                                                            # 保存当前输出，用于下一个快捷连接
        x = self.norm2(x)                                                       # 通过第二层的 LayerNorm
        x = self.ff(x)                                                          # 通过前馈网络
        x = self.drop_shortcut(x)                                               # 应用 Dropout
        x = x + shortcut                                                        # 将原始输入与当前输出相加，形成第二个快捷连接

        return x                                                                # 返回最终的输出



torch.manual_seed(123)
x = torch.rand(2, 4, 768)                                                       # Shape: [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)                                       #设定一个TransFormerBlock
output = block(x)                                                               #把input embedding通过block
print("Input shape:", x.shape)                                                  #(Input shape: torch.Size([2, 4, 768]))
print("Output shape:", output.shape)                                            #输入的形状和输出的形状是一样的（Output shape: torch.Size([2, 4, 768]))




"""
6.编写一个GPT模型

经过这个类处理后，相当于模型“完全理解”了输入的所有信息,
输出的最后一个向量值对应的就是可能输出的概率
"""

class GPTModel(nn.Module):                                                     # 定义 GPT 模型类，继承自 nn.Module
    def __init__(self, cfg):
        super().__init__()  
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])         # 定义词嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])     # 定义位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])                           # 定义嵌入后的 Dropout 层

        self.trf_blocks = nn.Sequential(                                       # 使用多个 Transformer 块组成的顺序模块
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]           # 根据层数创建多个 TransformerBlock
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])                            # 最后的 LayerNorm 层
        self.out_head = nn.Linear(                                             # 定义输出层，用于将隐藏层映射到词汇表维度
            cfg["emb_dim"], cfg["vocab_size"], bias=False                      # 输入为嵌入维度，输出为词汇表大小，无偏置
        )

    def forward(self, in_idx):                                                 # 定义前向传播逻辑
        batch_size, seq_len = in_idx.shape                                     # 获取输入张量的批量大小和序列长度
        tok_embeds = self.tok_emb(in_idx)                                      # 通过词嵌入层，得到词嵌入张量
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  
                                                                               # 通过位置嵌入层，生成序列中每个位置的嵌入
        x = tok_embeds + pos_embeds                                            # 将词嵌入和位置嵌入相加，形状 [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)                                                   # 对嵌入结果进行 Dropout
        x = self.trf_blocks(x)                                                 # 通过 Transformer 块
        x = self.final_norm(x)                                                 # 应用 LayerNorm
        logits = self.out_head(x)                                              # 将最后的隐藏层输出映射到词汇表大小，生成 logits
        return logits                                                          # 返回 logits，形状 [batch_size, num_tokens, vocab_size]

    

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)                                              #初始化一个GPT模型
out = model(batch)                                                             #传入一个批次数据
print("Input batch:\n", batch)                                                 # tensor([[6109, 3626, 6100,  345],
                                                                               # [6109, 1110, 6622,  257]])
print("\nOutput shape:", out.shape)                                            # (Output shape: torch.Size([2, 4, 50257]))
print(out)                                 
 
total_params = sum(p.numel() for p in model.parameters())                      #获取这个模型的全部参数个数
print(f"Total number of parameters: {total_params:,}")                         #(163,009,536)
total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())       #减去输出层的参数数量
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")      #(124,412,160)

#计算模型的内需求
total_size_bytes = total_params * 4                                            #每个参数float类型四个字节
# Convert to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")                      #(Total size of the model: 621.83 MB)


"""
7.生成文本

利用贪心解码策略（选择可能性最高的值）
"""


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 定义一个简单的文本生成函数
    # 参数：
    # - model: 用于生成文本的模型
    # - idx: 当前上下文中的 token 索引，形状为 (batch, n_tokens)
    # - max_new_tokens: 需要生成的新 token 的最大数量
    # - context_size: 支持的上下文最大长度

    for _ in range(max_new_tokens):                                           # 遍历每一个新生成的 token
        idx_cond = idx[:, -context_size:]                                     # 如果上下文长度超过 context_size，则截取最后 context_size 个 token

        # Get the predictions
        with torch.no_grad():                                                 # 禁用梯度计算，提高生成效率
            logits = model(idx_cond)                                          # 使用模型生成预测 logits，形状为 (batch, n_tokens, vocab_size)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]                                             # 只关注最后一个时间步的预测，形状变为 (batch, vocab_size)

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)                                # 将 logits 转化为概率分布，形状为 (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)                 # 找出概率最大的索引，形状为 (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)                               # 将生成的新 token 索引追加到序列中，更新形状为 (batch, n_tokens+1)

    return idx                                                                # 返回生成的完整序列，包括原始上下文和新生成的 token



start_context = "Hello, I am"                                                 #起始文本
encoded = tokenizer.encode(start_context)                                     #编码
print("encoded:", encoded)                                                    #(encoded: [15496, 11, 314, 716])
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)                          #(encoded_tensor.shape: torch.Size([1, 4]))

model.eval()                                                                  #将模型设置为评估模式，以禁用 Dropout 和 BatchNorm 等影响预测的训练行为
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,                                                         #产生6个新的文本
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)                                                         #(Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]]))
print("Output length:", len(out[0]))                                          #(Output length: 10)
         
decoded_text = tokenizer.decode(out.squeeze(0).tolist())                      #解码
print(decoded_text)                                                           #(Hello, I am Featureiman Byeswickattribute argue)