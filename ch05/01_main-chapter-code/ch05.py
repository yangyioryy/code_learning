"""
对未标记数据的预训练
"""

from importlib.metadata import version                                             #导入依赖库
pkgs = ["matplotlib", 
        "numpy", 
        "tiktoken", 
        "torch",
        "tensorflow" # For OpenAI's pretrained weights
       ] 
for p in pkgs:                                                                     #检测
    print(f"{p} version: {version(p)}")


"""
1.评估生成文本模型

交叉熵（Negative average log probability）： 在每个目标token的概率取对数，求平均，取负       让他尽可能地小 
困惑度 =交叉熵的额指数 可以视作模型每一步不确定的词汇可能数
"""

# 1.1 使用GPT生成文本
import torch
from previous_chapters import GPTModel                                            #导入上一节中的GPTModel
GPT_CONFIG_124M = {                                                               #模型基础参数设置
    "vocab_size": 50257,   # Vocabulary size 
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)                                                           #随机化种子
model = GPTModel(GPT_CONFIG_124M)                                                #创建模型对象
model.eval();  # Disable dropout during inference                                #设置为评估模式

import tiktoken                                                                  # 导入 tiktoken 库，用于处理文本与 token 的转换
from previous_chapters import generate_text_simple                               # 从 previous_chapters 导入 generate_text_simple 函数，用于生成文本

def text_to_token_ids(text, tokenizer):                                          # 定义一个函数，将文本转换为 token id
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})          # 使用 tokenizer 对文本进行编码，允许特殊字符 '<|endoftext|>' 
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)                          # 将编码后的 token 列表转为 tensor，并添加批次维度
    return encoded_tensor                                                        # 返回转换后的 tensor

def token_ids_to_text(token_ids, tokenizer):                                     # 定义一个函数，将 token ids 转回文本
    flat = token_ids.squeeze(0)                                                  # 去掉批次维度（即将 tensor 转为一维数组）
    return tokenizer.decode(flat.tolist())                                       # 将 token ids 解码为文本并返回

start_context = "Every effort moves you"                                         # 定义生成文本的起始上下文
tokenizer = tiktoken.get_encoding("gpt2")                                        # 获取 GPT-2 模型的 tokenizer 编码器
token_ids = generate_text_simple(                                                # 调用 generate_text_simple 函数生成文本
    model=model,                                                                 # 使用的模型
    idx=text_to_token_ids(start_context, tokenizer),                             # 将起始上下文文本转换为 token ids
    max_new_tokens=10,                                                           # 生成最多 10 个新 token
    context_size=GPT_CONFIG_124M["context_length"]                               # 设置上下文大小，使用 GPT_CONFIG_124M 中的 context_length
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))                 # 将生成的 token ids 转换回文本并输出（Every effort moves you rentingetic wasnم refres RexMeCHicular stren）



# 1.2 计算文本生成损失：交叉熵和困惑度

#初始化输入和目标输出
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]
targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]

with torch.no_grad():
    logits = model(inputs)                                                      #模型对输出的预测
probas = torch.softmax(logits, dim=-1)                                          #通过softmax层，转化为概率
print(probas.shape)                                                             # Shape: (batch_size, num_tokens, vocab_size)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)                          #获取预测概率最大的id
print("Token IDs:\n", token_ids)                                                #输出每个位置预测的下一个tokenid

#对预测结果进行解码
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")           #输入（Targets batch 1:  effort moves you）
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")#输出（Outputs batch 1:  Armed heNetflix）

text_idx = 0                                                                    #处理第一个文本
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]                
print("Text 1:", target_probas_1)                                               #(Text 1: tensor([7.4540e-05, 3.1061e-05, 1.1563e-05]))
text_idx = 1                                                                    #处理第二个文本
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]                
print("Text 2:", target_probas_2)                                               #(Text 2: tensor([1.0337e-05, 5.6776e-05, 4.7559e-06]))
#上面两个张量的每一个数据越接近1 说明离正确输出越近

# Compute logarithm of all token probabilities
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))          #将概率拼接后转化为对数值
print(log_probas)                                                              #(tensor([ -9.5042, -10.3796, -11.3677, -11.4798,  -9.7764, -12.2561]))

# Calculate the average probability for each token
avg_log_probas = torch.mean(log_probas)                                        #计算对数平均值
print(avg_log_probas)                                                          #(tensor(-10.7940))
neg_avg_log_probas = avg_log_probas * -1                                       #取负数让他最后尽可能的小到0
print(neg_avg_log_probas)                                                      #(tensor(10.7940))

# Logits have shape (batch_size, num_tokens, vocab_size)
print("Logits shape:", logits.shape)                                           #(Logits shape: torch.Size([2, 3, 50257]))
# Targets have shape (batch_size, num_tokens)
print("Targets shape:", targets.shape)                                         #(Targets shape: torch.Size([2, 3]))

#接下来是使用pytorch自带的交叉熵函数
logits_flat = logits.flatten(0, 1)                                             #张平
targets_flat = targets.flatten()                                               #张平
print("Flattened logits:", logits_flat.shape)                                  #(Flattened logits: torch.Size([6, 50257]))
print("Flattened targets:", targets_flat.shape)                                #(Flattened targets: torch.Size([6]))
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)            #将展平后的结果传入交叉熵函数
print(loss)                                                                    #(tensor(10.7940))

perplexity = torch.exp(loss)                                                   #困惑度=交叉熵的指数
print(perplexity)                                                              #(tensor(48725.8203))



# 1.3 计算训练和验证集的损失 
import os
import urllib.request 
file_path = "the-verdict.txt"                                                  #一个较小的数据集
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')                  
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()                                              #读取数据集中的内容
# First 100 characters                                                       #打印前100个元素
print(text_data[:99])                                                        #(I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no )
# Last 100 characters                                                        #打印最后100个元素
print(text_data[-99:])                                                       #(it for me! The Strouds stand alone, and happen once--but there's no exterminating our kind of art.")
total_characters = len(text_data)                                            #总词汇数
total_tokens = len(tokenizer.encode(text_data))                              #词汇表大小
print("Characters:", total_characters)                                       #(Characters: 20479)
print("Tokens:", total_tokens)                                               #(Tokens: 5145)


from previous_chapters import create_dataloader_v1                           # 从 previous_chapters 模块导入 create_dataloader_v1 函数

# Train/validation ratio
train_ratio = 0.90                                                           # 定义训练集与总数据的比例为 90%
split_idx = int(train_ratio * len(text_data))                                # 根据比例计算训练集与验证集的分界索引
train_data = text_data[:split_idx]                                           # 切片获取训练集数据
val_data = text_data[split_idx:]                                             # 切片获取验证集数据

torch.manual_seed(123)                                                       # 设置随机种子以确保可复现性
train_loader = create_dataloader_v1(                                         # 创建训练集数据加载器
    train_data,                                                              # 输入训练集数据
    batch_size=2,                                                            # 设置批量大小为 2
    max_length=GPT_CONFIG_124M["context_length"],                            # 每个样本的最大长度为模型的上下文长度
    stride=GPT_CONFIG_124M["context_length"],                                # 样本步幅设置为上下文长度
    drop_last=True,                                                          # 如果最后一个批次不完整则丢弃
    shuffle=True,                                                            # 对数据进行随机打乱
    num_workers=0                                                            # 使用的子线程数量为 0，表示在主线程中加载数据
)

val_loader = create_dataloader_v1(                                           # 创建验证集数据加载器
    val_data,                                                                # 输入验证集数据
    batch_size=2,                                                            # 设置批量大小为 2
    max_length=GPT_CONFIG_124M["context_length"],                            # 每个样本的最大长度为模型的上下文长度
    stride=GPT_CONFIG_124M["context_length"],                                # 样本步幅设置为上下文长度
    drop_last=False,                                                         # 如果最后一个批次不完整则保留
    shuffle=False,                                                           # 不对数据进行随机打乱
    num_workers=0                                                            # 使用的子线程数量为 0，表示在主线程中加载数据
)

# Sanity check
if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:         # 检查训练集 token 总数是否小于上下文长度
    print("Not enough tokens for the training loader. "                      # 如果不够，打印警告信息
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "  
          "increase the `training_ratio`")  

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:      # 检查验证集 token 总数是否小于上下文长度
    print("Not enough tokens for the validation loader. "                   # 如果不够，打印警告信息
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "  
          "decrease the `training_ratio`")  
    
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)                                                 # 打印训练集中数据形状
print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)                                                 # 打印验证集数据形状

train_tokens = 0                          
for input_batch, target_batch in train_loader:           
    train_tokens += input_batch.numel()                                     # 计算训练集的token数
val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()                                       # 计算验证集中的token数
print("Training tokens:", train_tokens)                                     # （Training tokens: 4608）
print("Validation tokens:", val_tokens)                                     # (Validation tokens: 512)
print("All tokens:", train_tokens + val_tokens)                             # (All tokens: 5120)


def calc_loss_batch(input_batch, target_batch, model, device):              # 定义函数，计算一个批次的损失
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入批次和目标批次的数据移动到指定设备（如 GPU 或 CPU）
    logits = model(input_batch)                                             # 使用模型对输入批次进行前向传播，得到输出 logits
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())  # 计算交叉熵损失，将 logits 展平成二维，目标展平成一维
    return loss                                                             # 返回计算的损失

def calc_loss_loader(data_loader, model, device, num_batches=None):         # 定义函数，计算整个数据加载器上的平均损失
    total_loss = 0.                                                         # 初始化总损失为 0
    if len(data_loader) == 0:                                               # 如果数据加载器为空
        return float("nan")                                                 # 返回非数字 NaN 表示无效的结果
    elif num_batches is None:                                               # 如果未指定 num_batches
        num_batches = len(data_loader)                                      # 将数据加载器的总批次数赋值给 num_batches
    else:  
        num_batches = min(num_batches, len(data_loader))                    # 将 num_batches 的值限制为数据加载器的总批次数
    for i, (input_batch, target_batch) in enumerate(data_loader):  
        if i < num_batches:                                                 # 调用 calc_loss_batch 计算当前批次的损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)  
            total_loss += loss.item()                                       # 累加当前批次的损失值（转换为 Python 的浮点数）
        else:  
            break                       
    return total_loss / num_batches                                         # 返回平均损失（总损失除以批次数）


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.to(device)                                                            
torch.manual_seed(123) 

with torch.no_grad():                                                       # 此处先禁用梯度
    train_loss = calc_loss_loader(train_loader, model, device)              # 计算训练集上的平均损失
    val_loss = calc_loss_loader(val_loader, model, device)                  # 计算验证集上的平均损失

#这里的训练集和验证集实际上是没有区别的，因为模型并没有到训练集上进行一个训练
print("Training loss:", train_loss)                                         #（Training loss: 10.987583266364204）
print("Validation loss:", val_loss)                                         #（Training loss: 10.987583266364204）



"""
2.训练大模型

实现一个简单的训练逻辑，通过梯度下降来不断减少损失，从而更新权重
"""

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):     # 定义简单的模型训练函数
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []                # 初始化列表，用于记录训练损失、验证损失和已处理的token数量
    tokens_seen, global_step = 0, -1                                        # 初始化已处理的token数量为0，全局步数为-1

    # Main training loop
    for epoch in range(num_epochs):                                         # 遍历每个训练周期
        model.train()                                                       # 设置模型为训练模式
        
        for input_batch, target_batch in train_loader:                      # 遍历训练数据加载器中的每个批次
            optimizer.zero_grad()                                           # 清除上一批次的梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)# 计算当前批次的损失，训练的过程让损失逐渐变小就行
            loss.backward()                                                 # 反向传播计算梯度
            optimizer.step()                                                # 使用梯度更新模型权重
            tokens_seen += input_batch.numel()                              # 累加已处理的token数量
            global_step += 1                                                # 增加全局步数

            # Optional evaluation step
            if global_step % eval_freq == 0:                                # 每经过指定步数进行一次评估
                train_loss, val_loss = evaluate_model(                      # 调用评估函数，获取训练和验证损失
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)                             # 将训练损失记录到列表中
                val_losses.append(val_loss)                                 # 将验证损失记录到列表中
                track_tokens_seen.append(tokens_seen)                       # 将已处理的token数量记录到列表中
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")  
                                                                            # 打印当前周期、步数、训练损失和验证损失

        # Print a sample text after each epoch
        generate_and_print_sample(                                          # 在每个周期结束后生成并打印样本文本
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen                      # 返回训练损失、验证损失和已处理的token数量


def evaluate_model(model, train_loader, val_loader, device, eval_iter):     # 定义模型评估函数
    model.eval()                                                            # 设置模型为评估模式
    with torch.no_grad():                                                   # 禁用梯度计算以加速评估
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  
                                                                            # 在训练加载器上计算指定批次数的损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  
                                                                            # 在验证加载器上计算指定批次数的损失
    model.train()                                                           # 恢复模型为训练模式
    return train_loss, val_loss                                             # 返回训练损失和验证损失


def generate_and_print_sample(model, tokenizer, device, start_context):     # 定义生成和打印样本文本的函数
    model.eval()                                                            # 设置模型为评估模式
    context_size = model.pos_emb.weight.shape[0]                            # 获取位置嵌入的上下文大小
    encoded = text_to_token_ids(start_context, tokenizer).to(device)        # 将起始上下文转换为token ID并移动到指定设备上
    with torch.no_grad():                                                   # 禁用梯度计算以加速文本生成
        token_ids = generate_text_simple(                                   # 调用文本生成函数
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)                  # 将生成的token ID解码为文本
    print(decoded_text.replace("\n", " "))                                  # 打印生成的文本，并将换行符替换为空格以简化显示
    model.train()                                                           # 恢复模型为训练模式


torch.manual_seed(123)                                                      # 设置随机种子以保证结果的可重复性
model = GPTModel(GPT_CONFIG_124M)                                           # 初始化 GPT 模型，使用指定的配置参数
model.to(device)                                                            # 将模型移动到指定设备（如 GPU 或 CPU）
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)  
                                                                            # 定义优化器，使用 AdamW 优化算法，设置学习率和权重衰减系数

num_epochs = 10                                                             # 定义训练周期数为 10
train_losses, val_losses, tokens_seen = train_model_simple(  
                                                                            # 调用训练函数并接收返回值，包括训练损失、验证损失和已处理的 token 数量
    model, train_loader, val_loader, optimizer, device,                     # 输入模型、数据加载器、优化器和设备
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,                        # 指定训练周期数，每隔 5 步评估一次，评估时的批次数为 5
    start_context="Every effort moves you", tokenizer=tokenizer             # 设置初始上下文和分词器
)



import matplotlib.pyplot as plt                                             # 导入 Matplotlib 库用于绘图
from matplotlib.ticker import MaxNLocator                                   # 导入工具以确保 x 轴标签为整数

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):        # 定义一个函数用于绘制训练和验证损失图

    fig, ax1 = plt.subplots(figsize=(5, 3))                                 # 创建一个绘图窗口，主轴用于显示训练和验证损失曲线，指定图形大小为 5x3
    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")              # 绘制训练损失曲线，横轴为经历的训练周期数
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")    # 绘制验证损失曲线，使用虚点线，横轴为经历的训练周期数
    ax1.set_xlabel("Epochs")                                                # 设置主 x 轴标签为 “Epochs”
    ax1.set_ylabel("Loss")                                                  # 设置 y 轴标签为 “Loss”
    ax1.legend(loc="upper right")                                           # 显示图例，位置设在右上角
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  
                                                                            # 将主 x 轴的刻度标签设置为整数，以便更清晰地显示训练周期
    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()                                                       # 创建一个共享 y 轴的第二 x 轴，用于显示处理的 token 数量
    ax2.plot(tokens_seen, train_losses, alpha=0)                            # 绘制一个不可见的曲线（透明度设置为 0）以对齐 token 数的刻度
    ax2.set_xlabel("Tokens seen")                                           # 设置第二 x 轴标签为 “Tokens seen”

    fig.tight_layout()                                                      # 调整图形布局，以避免标签和图形重叠
    plt.savefig("loss-plot.pdf")                                            # 将图形保存为 PDF 文件，文件名为 “loss-plot.pdf”
    plt.show()                                                              # 显示绘制的图形

# 调用绘图函数，绘制损失曲线图并显示
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))  
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)  



"""
3.控制随机性的解码策略，让模型不是每次都选择概率最大的生成

3.1 temperature scaling 把logits除以一个大于0的数temperature

3.2 top-k sampling 只保留概率前k大的值
"""

model.to("cpu")                                                            # 将模型移动到 CPU 上进行推理
model.eval()                                                               # 将模型设置为评估模式，禁用 dropout 等训练特性
tokenizer = tiktoken.get_encoding("gpt2")                                  # 加载 GPT-2 的分词器以处理文本

token_ids = generate_text_simple(  
    model=model,                                                           # 指定使用的模型
    idx=text_to_token_ids("Every effort moves you", tokenizer),            # 将初始文本 "Every effort moves you" 转换为 token ID，并作为输入上下文
    max_new_tokens=25,                                                     # 指定生成的最大新 token 数为 25
    context_size=GPT_CONFIG_124M["context_length"]                         # 指定上下文长度（即模型可处理的最大 token 长度）
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))  



# 3.1 Temperature scaling

vocab = {                                                                  # 指定一个小的词汇表字典
    "closer": 0,      
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 
     
inverse_vocab = {v: k for k, v in vocab.items()}                          # 反向
# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(                                         #对于输入放回的概率分布
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)
probas = torch.softmax(next_token_logits, dim=0)                          #通过softmax函数转化为取某个token的概率
next_token_id = torch.argmax(probas).item()                               #之前做法选择概率最大的
# The next generated token is then as follows:
print(inverse_vocab[next_token_id])                                       #解码功能（forward）



def print_sampled_tokens(probas):
    torch.manual_seed(123)                                                # 设置随机数种子
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]#调用multinomial函数从probas中按概率进行抽样
    sampled_ids = torch.bincount(torch.tensor(sample))                    # 使用bitcount函数统计每个sample中各元素的次数
    for i, freq in enumerate(sampled_ids):                                # 遍历打印
        print(f"{freq} x {inverse_vocab[i]}")
print_sampled_tokens(probas)


def softmax_with_temperature(logits, temperature):                        # 带temperature的softmax函数
    scaled_logits = logits / temperature                                  # 把logits 除以temperate
    return torch.softmax(scaled_logits, dim=0)                            # 再进行一个softmax函数

temperatures = [1, 0.1, 5]                                                # Original, higher confidence, and lower confidence
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures] #不同temperature下的答案

x = torch.arange(len(vocab))                                              # 创建一个张量 `x`，其值为从 0 到 vocab 长度的整数序列，用作 x 轴位置
bar_width = 0.15                                                          # 设置条形图的宽度为 0.15
fig, ax = plt.subplots(figsize=(5, 3))                                    # 创建一个大小为 5x3 英寸的图形对象和坐标轴对象

for i, T in enumerate(temperatures):                                      # 遍历 temperatures 中的每个温度值，并返回索引 i 和对应的温度 T
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')  
    # 绘制条形图，`x + i * bar_width` 为每组条形图的偏移位置，`scaled_probas[i]` 为该组的概率，`label` 用于图例

ax.set_ylabel('Probability')                                              # 设置 y 轴标签为 'Probability'
ax.set_xticks(x)                                                          # 设置 x 轴的刻度位置为 `x`，即词汇表中的每个单词的位置
ax.set_xticklabels(vocab.keys(), rotation=90)                             # 设置 x 轴的刻度标签为 `vocab` 的键（即词汇表的单词），并将标签旋转 90 度
ax.legend()                                                               # 显示图例
plt.tight_layout()                                                        # 调整布局，使元素适应图形区域
plt.savefig("temperature-plot.pdf")                                       # 将图形保存为 "temperature-plot.pdf"
plt.show()                                                                # 显示图形

print_sampled_tokens(scaled_probas[1])                                    #（对temperature=0.1下的答案分布）
print_sampled_tokens(scaled_probas[2])                                    #(对temperature=5下的答案分布)


# 3.2 Top-k sampling

top_k = 3                                                                 # top-k设置为3
top_logits, top_pos = torch.topk(next_token_logits, top_k)                # 直接调用torch的topk函数就可
print("Top logits:", top_logits)                                          # （Top logits: tensor([6.7500, 6.2800, 4.5100])）
print("Top positions:", top_pos)                                          # （Top positions: tensor([3, 7, 0])）

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],                         # 选择元素，条件为 `next_token_logits` 中的元素是否小于 `top_logits[-1]`（top_logits 列表中的最后一个元素）
    input=torch.tensor(float("-inf")),                                    # 如果条件为真，则将该位置的 logits 设置为负无穷大（表示该 token 不会被选择）
    other=next_token_logits                                               # # 否则，保持 `next_token_logits` 的原值
)

print(new_logits)                                                         # 没被选出的元素全部设置为最小值(tensor([4.5100,   -inf,   -inf, 6.7500,   -inf,   -inf,   -inf, 6.2800,   -inf]))

new_logits = torch.full_like(                                             # create tensor containing -inf values
   next_token_logits, -torch.inf
)   
new_logits[top_pos] = next_token_logits[top_pos]                          # copy top k values into the -inf tensor
topk_probas = torch.softmax(new_logits, dim=0)                            #再进行softmax函数
print(topk_probas)                                                        # （tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])）

# 3.3 修改文本生成函数
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):                                       # For-loop is the same as before: Get logits, and only focus on last time step
        idx_cond = idx[:, -context_size:]                                 # 获取最后context_size长度的tokens作为模型的输入
        with torch.no_grad():
            logits = model(idx_cond)                                      # 使用模型生成logits
        logits = logits[:, -1, :]                                         # 只保留最后一个时间步

        if top_k is not None:                                             # New: Filter logits with top_k sampling
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]                                   # 将小于第 k 大的值的 logits 设置为负无穷，表示不选这些 token
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        
        if temperature > 0.0:                                             # New: Apply temperature scaling
            logits = logits / temperature                                 # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)                         # (batch_size, context_len)
            idx_next = torch.multinomial(probs, num_samples=1)            # Sample from the distribution

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)         # (batch_size, 1)

        if idx_next == eos_id:                                            # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)                           # (batch_size, num_tokens+1)

    return idx

torch.manual_seed(123)                                                    # 设置随机数种子
token_ids = generate(                                                     # 利用新的generate函数
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))          #（ Every effort moves you know began to go a little wild--I was such a good; and）



"""
4.在朋友torch中加载和保存模型的训练过程中的权重

torch.save 函数应用于 .state_dict() 方法来保存模型权重
"""

torch.save(model.state_dict(), "model.pth")                               # 保存权重

model = GPTModel(GPT_CONFIG_124M)                                         # 再初始化一个模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     #设定训练的设备
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))#直接把模型保存的权重加载到另一个模型中
model.eval();                                                             # 设置为评估模式

torch.save({                                                              # 将模型和优化器的状态字典保存到一个文件中
    "model_state_dict": model.state_dict(),                               # 保存模型的参数状态字典
    "optimizer_state_dict": optimizer.state_dict(),                       # 保存优化器的参数状态字典
    },
    "model_and_optimizer.pth"                                             # 指定保存的文件名
)

checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)     # 加载保存的文件，weights_only指定是否只加载权重
model = GPTModel(GPT_CONFIG_124M)                                         # 初始化一个GPT模型实例
model.load_state_dict(checkpoint["model_state_dict"])                     # 将保存的模型参数加载到模型实例中
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)  # 使用AdamW优化器初始化
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])             # 加载保存的优化器参数到优化器实例中
model.train()                                                             # 将模型设置为训练模式


"""
5.使用OpenAI的预训练权重

使用gpt2的开源的预训练的权重参数,下载之后还需要手动进行相应的配置
"""

# pip install tensorflow tqdm                                             # tensorflow是openai用于加载权重的库，tqdm是显示进度条的库
print("TensorFlow version:", version("tensorflow"))                       # 检测
print("tqdm version:", version("tqdm"))

# Relative import from the gpt_download.py contained in this folder
from gpt_download import download_and_load_gpt2                           # 其他一些依赖库
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")#开始导入gpt2的权重参数
print("Settings:", settings)                                              #（Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}）
print("Parameter dictionary keys:", params.keys())                        # 模型参数的key（Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])）
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)   # （Token embedding weight tensor dimensions: (50257, 768)）


# 定义模型配置字典，用于紧凑地管理不同GPT模型的参数设置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # gpt2-small模型的配置
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},# gpt2-medium模型的配置
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, # gpt2-large模型的配置
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},   # gpt2-xl模型的配置
}

# 复制基础配置并更新为指定模型的具体设置
model_name = "gpt2-small (124M)"                                           # 示例使用gpt2-small模型
NEW_CONFIG = GPT_CONFIG_124M.copy()                                        # 复制基础配置
NEW_CONFIG.update(model_configs[model_name])                               # 更新为指定模型的配置
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})              # 增加上下文长度和QKV偏置设置
gpt = GPTModel(NEW_CONFIG)                                                 # 使用更新后的配置初始化GPT模型
gpt.eval()                                                                 # 将模型设置为推理模式

# 定义参数赋值函数
def assign(left, right):  
    if left.shape != right.shape:                                          # 检查左右参数形状是否匹配
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")  # 报错不匹配的情况
    return torch.nn.Parameter(torch.tensor(right))                         # 转换为torch参数并返回

import numpy as np  # 导入numpy库

# 定义将参数加载到GPT模型中的函数
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])          # 加载位置嵌入参数
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])          # 加载词嵌入参数

    for b in range(len(params["blocks"])):                                  # 遍历每个transformer block
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)       # 分割查询、键和值的权重
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)                    # 加载查询权重
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)                      # 加载键权重
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)                    # 加载值权重

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)       # 分割查询、键和值的偏置
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)                        # 加载查询偏置
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)                          # 加载键偏置
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)                        # 加载值偏置

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)                   # 加载输出投影权重
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])                     # 加载输出投影偏置

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)                       # 加载前馈网络第一个全连接层权重
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])                         # 加载第一个全连接层偏置
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)                     # 加载第二个全连接层权重
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])                       # 加载第二个全连接层偏置

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])                                # 加载第一个规范化层的缩放参数
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])                                # 加载第一个规范化层的偏移参数
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])                                # 加载第二个规范化层的缩放参数
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])                                # 加载第二个规范化层的偏移参数

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])         # 加载最终规范化层的缩放参数
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])         # 加载最终规范化层的偏移参数
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])         # 加载输出层权重

# 将权重加载到GPT模型中
load_weights_into_gpt(gpt, params)  
gpt.to(device)                                                               # 将模型加载到设备上（如GPU或CPU）
torch.manual_seed(123)                                                       # 设置随机种子以保证生成结果可重复
token_ids = generate(                                                        # 调用生成函数
    model=gpt,                                                               # 使用加载了权重的GPT模型
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),   # 将初始文本转换为token ID
    max_new_tokens=25,                                                       # 最大生成的token数
    context_size=NEW_CONFIG["context_length"],                               # 使用模型的上下文长度
    top_k=50,                                                                # 使用Top-K采样
    temperature=1.5                                                          # 设置温度参数影响多样性
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))             #(Every effort moves you as far as the hand can go until the end of your turn unless something happens)
