"""
微调一个指令跟随模型
"""

from importlib.metadata import version                                              # 导入依赖库                                       
pkgs = [                                                        
    "matplotlib",  # Plotting library
    "tiktoken",    # Tokenizer
    "torch",       # Deep learning library
    "tqdm",        # Progress bar
    "tensorflow",  # For OpenAI's pretrained weights
]
for p in pkgs:                                                                      # 检测版本信息
    print(f"{p} version: {version(p)}")


"""
1.导入指令跟随的微调数据集

数据集处理上，在输入前拼接一个提示
"""

import json                                                                         # 相关依赖库的导入
import os
import urllib

def download_and_load_file(file_path, url):                                         # 下载并加载文件
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)                                                   # 将下载的文件写入对应文件
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)                                                      # 返回json格式数据

    return data


file_path = "instruction-data.json"                                                 # 指定数据的存放地址
url = (                                                                             # 数据下载地址
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch" 
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)                                       # 下载并读取数据
print("Number of entries:", len(data))                                              # （Number of entries: 1100）
print("Example entry:\n", data[50])                                                 # 字典形式的数据
print("Another example entry:\n", data[999])                                        # key的值有instruction input和output，input可以为空


def format_input(entry):
    instruction_text = (                                                            # 对数据格式化输入
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""      
    return instruction_text + input_text                                            # 把提示和输入进行拼接

model_input = format_input(data[50])                                                # 有input的数据
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)
model_input = format_input(data[999])                                               # 无input的数据
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)


train_portion = int(len(data) * 0.85)                                               # 85% for training
test_portion = int(len(data) * 0.1)                                                 # 10% for testing
val_portion = len(data) - train_portion - test_portion                              # Remaining 5% for validation

train_data = data[:train_portion]                                                   # 切片划分
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]


"""
2. 把数据组织为训练批次，数据预处理

这里是把输入和输出格式化之后作为一个整体给模型进行训练，让它接下一个token
这里数据长度统一只在同一个批次中进行了统一

用-100代替输出中的特殊字符，在利用朋友torch的函数计算交叉熵时，会自动忽略含-100的测试用例带来的损失
"""

import torch
from torch.utils.data import Dataset
class InstructionDataset(Dataset):                                                 # 创建一个dataset
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []                                                    # 编码后文本
        for entry in data:                                                         # 遍历每一个用例
            instruction_plus_input = format_input(entry)                           # 格式化输入
            response_text = f"\n\n### Response:\n{entry['output']}"                # 输出文本
            full_text = instruction_plus_input + response_text                     # 完整文本
            self.encoded_texts.append(                                             # 对完整文本进行编码
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]                                           # 返回指定索引的编码文本

    def __len__(self):
        return len(self.data)                                                      # 返回数据的长度


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")                                          # 引入编码器
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))        # 允许特殊字符（[50256]）



def custom_collate_draft_1(
    batch,                                                                         # 输入的批次数据
    pad_token_id=50256,                                                            # 填充标记的ID，默认为50256
    device="cpu"                                                                   # 设备类型，默认为CPU
):
    # 找到批次中最长的序列，并将最大长度增加1，这样可以在下面加入一个额外的填充标记
    batch_max_length = max(len(item) + 1 for item in batch)                        # 获取批次中最长的序列长度，并加1

    # 用于存放处理后的输入列表
    inputs_lst = []                                                                # 初始化空列表用于存储处理后的数据

    for item in batch:                                                             # 遍历批次中的每个序列
        new_item = item.copy()                                                     # 复制当前序列
        # 在序列末尾添加一个<|endoftext|>标记（填充标记）
        new_item += [pad_token_id]                                                 # 在当前序列末尾添加填充标记
        # 将序列填充到与最大序列长度一致
        padded = (
            new_item + [pad_token_id] *                                            # 拼接填充标记，使长度达到最大值
            (batch_max_length - len(new_item))                                     # 计算需要填充的数量
        )
        # 通过padded[:-1]移除之前额外添加的填充标记，这个额外的填充标记在后续处理中会被使用
        inputs = torch.tensor(padded[:-1])                                         # 将填充后的序列转换为Tensor，并去掉最后一个填充标记
        inputs_lst.append(inputs)                                                  # 将处理后的输入序列加入列表

    # 将所有输入序列的Tensor合并为一个批次，并将其移动到指定的设备（如GPU） 
    inputs_tensor = torch.stack(inputs_lst).to(device)                             # 使用torch.stack将多个Tensor堆叠成一个批次，并将其转移到目标设备
    return inputs_tensor                                                           # 返回处理后的输入批次Tensor

# 定义三个输入序列
inputs_1 = [0, 1, 2, 3, 4]                                                       
inputs_2 = [5, 6]  
inputs_3 = [7, 8, 9] 

# 将三个输入序列组成一个批次
batch = (
    inputs_1,  
    inputs_2,  
    inputs_3  
)

# 调用自定义的collate函数并打印输出
print(custom_collate_draft_1(batch))                                             # 每个输入都和该批次最长输入一样长

def custom_collate_draft_2(
    batch,                                                                       # 输入的批次数据
    pad_token_id=50256,                                                          # 填充标记的ID，默认为50256
    device="cpu"                                                                 # 设备类型，默认为CPU
):
    batch_max_length = max(len(item) + 1 for item in batch) 

    # 用于存放处理后的输入和目标列表
    inputs_lst, targets_lst = [], []  # 初始化两个空列表分别用于存放输入和目标

    for item in batch:                                                          # 遍历批次中的每个序列
        new_item = item.copy()                                                  # 复制当前序列
        # 在序列末尾添加一个<|endoftext|>标记（填充标记）
        new_item += [pad_token_id]                                              # 在当前序列末尾添加填充标记
        # 将序列填充到与最大序列长度一致
        padded = (
            new_item + [pad_token_id] *                                         # 拼接填充标记，使长度达到最大值
            (batch_max_length - len(new_item))                                  # 计算需要填充的数量
        )
        inputs = torch.tensor(padded[:-1])                                      # 将填充后的序列转换为Tensor，去掉最后一个填充标记作为输入
        targets = torch.tensor(padded[1:])                                      # 将填充后的序列转换为Tensor，去掉第一个填充标记作为目标（shift+1）
        inputs_lst.append(inputs)                                               # 将输入序列加入列表
        targets_lst.append(targets)                                             # 将目标序列加入列表

    # 将所有输入序列和目标序列的Tensor合并为一个批次，并将其转移到指定的设备（如GPU）
    inputs_tensor = torch.stack(inputs_lst).to(device)                          # 使用torch.stack将多个输入Tensor堆叠成一个批次，并转移到目标设备
    targets_tensor = torch.stack(targets_lst).to(device)                        # 使用torch.stack将多个目标Tensor堆叠成一个批次，并转移到目标设备
    return inputs_tensor, targets_tensor                                        # 返回处理后的输入和目标批次Tensor

inputs, targets = custom_collate_draft_2(batch)                                 # 调用collate函数处理批次数据
print(inputs)                                                                   # 打印输入Tensor
print(targets)                                                                  # 打印目标Tensor

# ---------------------------------------------------------------

def custom_collate_fn(
    batch,  
    pad_token_id=50256,  
    ignore_index=-100,                                                          # 用于替换填充标记的目标索引，默认为-100
    allowed_max_length=None,                                                    # 可选的最大序列长度
    device="cpu"  
):
    batch_max_length = max(len(item) + 1 for item in batch)  

    inputs_lst, targets_lst = [], []  

    for item in batch:  
        new_item = item.copy()  
        new_item += [pad_token_id]  
        padded = (
            new_item + [pad_token_id] *  
            (batch_max_length - len(new_item))  
        )
        inputs = torch.tensor(padded[:-1]) 
        targets = torch.tensor(padded[1:])  

        # 新增：将目标中的填充标记以ignore_index替代，避免计算损失时影响梯度
        mask = targets == pad_token_id                                       # 找到目标中等于pad_token_id的位置
        indices = torch.nonzero(mask).squeeze()                              # 获取所有填充标记的索引位置
        if indices.numel() > 1:                                              # 如果有多个填充标记，忽略第一个填充标记
            targets[indices[1:]] = ignore_index                              # 将除第一个外的填充标记替换为ignore_index

        # 新增：可选地对序列进行截断，以保证不超过最大序列长度
        if allowed_max_length is not None:                                   # 如果设置了最大长度
            inputs = inputs[:allowed_max_length]                             # 截断输入序列
            targets = targets[:allowed_max_length]                           # 截断目标序列

        inputs_lst.append(inputs)  
        targets_lst.append(targets) 

    inputs_tensor = torch.stack(inputs_lst).to(device)  
    targets_tensor = torch.stack(targets_lst).to(device)  
    return inputs_tensor, targets_tensor 

inputs, targets = custom_collate_fn(batch)                                  # 调用collate函数处理批次数据
print(inputs)                                                               # 打印输入Tensor
print(targets)                                                              # 打印目标Tensor

logits_1 = torch.tensor(
    [[-1.0, 1.0],                                                           # 1st training example
     [-0.5, 1.5]]                                                           # 2nd training example
)
targets_1 = torch.tensor([0, 1])                                            # 输出
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)             # 计算交叉熵损失
print(loss_1)                                                               # （tensor(1.1269)）

logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]                                                          # 多添加一个测试用例
)
targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)                                                              # （tensor(0.7936)）

targets_3 = torch.tensor([0, 1, -100])                                     # 把某一个输出用-100代替
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)            # 计算交叉熵
print(loss_3)                                                              # （tensor(1.1269)）
print("loss_1 == loss_3:", loss_1 == loss_3)                               # （loss_1 == loss_3: tensor(True)）



"""
3.创建数据加载器，转化为可迭代对象
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     # 检测是否有GPU设备
print("Device:", device)                                                  # 打印当前的设备

from functools import partial
customized_collate_fn = partial(                                          # 相关数据的填充
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

from torch.utils.data import DataLoader                                   # 导入依赖库

num_workers = 0                                                           # 设置加载数据时的线程数，0表示不使用多线程
batch_size = 8                                                            # 设置每个批次的大小为8
torch.manual_seed(123)                                                    # 设置随机种子，确保结果的可复现性
train_dataset = InstructionDataset(train_data, tokenizer)                 # 创建训练集数据集对象，传入训练数据和分词器
train_loader = DataLoader(                                                # 创建训练集数据加载器
    train_dataset,                                                        # 使用的训练集数据集对象
    batch_size=batch_size,                                                # 设置每个批次的数据量
    collate_fn=customized_collate_fn,                                     # 设置自定义的collate函数用于处理批次数据
    shuffle=True,                                                         # 是否打乱数据顺序
    drop_last=True,                                                       # 如果批次的大小不够一个完整的batch，则丢弃该批次
    num_workers=num_workers                                               # 使用的加载数据的工作线程数
)

val_dataset = InstructionDataset(val_data, tokenizer)                     # 创建验证集数据集对象，传入验证数据和分词器
val_loader = DataLoader(                                                  # 创建验证集数据加载器
    val_dataset,  
    batch_size=batch_size,  
    collate_fn=customized_collate_fn, 
    shuffle=False,                                                        # 验证集不需要打乱数据顺序
    drop_last=False,                                                      # 不丢弃最后一个批次
    num_workers=num_workers  
)

test_dataset = InstructionDataset(test_data, tokenizer)                   # 创建测试集数据集对象，传入测试数据和分词器
test_loader = DataLoader(                                                 # 创建测试集数据加载器
    test_dataset,
    batch_size=batch_size,  
    collate_fn=customized_collate_fn,  
    shuffle=False, 
    drop_last=False,  
    num_workers=num_workers  
)

print("Train loader:")                                                    # 输出训练数据加载器的信息
for inputs, targets in train_loader:                                      # 遍历训练集数据加载器，获取每个批次的数据
    print(inputs.shape, targets.shape)                                    # batchsize都为8，不同批次长度不一样
print(inputs[0])                                                          # 打印第一个批次输入的内容（特殊标记为50256）
print(targets[0])                                                         # 打印第一个批次目标的内容（特殊标记为-100）


"""
4. 加载预训练模型

这里使用了中型GPT2模型配置
"""


from gpt_download import download_and_load_gpt2                           # 导入下载并加载GPT-2模型的函数
from previous_chapters import GPTModel, load_weights_into_gpt             # 导入自定义的GPT模型和加载权重的函数

BASE_CONFIG = {
    "vocab_size": 50257,                                                  # 词汇表大小
    "context_length": 1024,                                               # 上下文长度
    "drop_rate": 0.0,                                                     # 丢弃率（dropout）
    "qkv_bias": True                                                      # 是否使用Query-Key-Value的偏置
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # 小型GPT2模型配置
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, # 中型GPT2模型配置
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, # 大型GPT2模型配置
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},   # 超大型GPT2模型配置
}

CHOOSE_MODEL = "gpt2-medium (355M)"                                       # 选择使用的模型大小

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])                           # 更新基础配置，选择的模型会覆盖默认设置

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")          # 获取模型的大小，例如"355M"
settings, params = download_and_load_gpt2(                                # 下载并加载GPT-2模型的权重和配置
    model_size=model_size,                                                # 模型大小
    models_dir="gpt2"                                                     # 模型存放的目录
)

model = GPTModel(BASE_CONFIG)                                             # 使用更新后的配置初始化模型
load_weights_into_gpt(model, params)                                      # 将下载的权重加载到模型中
model.eval();                                                             # 设置模型为评估模式，禁用dropout等训练时特有的操作

torch.manual_seed(123)                                                    # 设置随机种子，确保结果的可复现性
input_text = format_input(val_data[0])                                    # 格式化输入文本，val_data[0]为输入文本的第一个元素
print(input_text)                                                         # 打印格式化后的输入文本

from previous_chapters import (
    generate,                                                             # 导入生成文本的函数
    text_to_token_ids,                                                    # 导入文本转为token id的函数
    token_ids_to_text                                                     # 导入token id转为文本的函数
)

token_ids = generate(                                                     # 生成文本
    model=model,                                                          # 使用的GPT模型
    idx=text_to_token_ids(input_text, tokenizer),                         # 将输入文本转为token ids
    max_new_tokens=35,                                                    # 最大生成新tokens的数量
    context_size=BASE_CONFIG["context_length"],                           # 上下文大小
    eos_id=50256,                                                         # 结束符ID
)
generated_text = token_ids_to_text(token_ids, tokenizer)                  # 将生成的token ids转换回文本

response_text = (
    generated_text[len(input_text):]                                      # 去掉输入文本部分，保留生成的部分
    .replace("### Response:", "")                                         # 移除生成文本中的标记
    .strip()                                                              # 去除多余的空白字符
)
print(response_text)                                                      # 打印生成的回复文本
#此时的模型并没有指令跟随的能力



"""
5.使用指令跟随的数据集进行微调

这里似乎是对全参数进行了微调
"""

from previous_chapters import (
    calc_loss_loader,                                                    # 损失函数
    train_model_simple                                                   # 训练函数
)


# 打印微调前训练集和验证集上的损失
model.to(device)
torch.manual_seed(123)
with torch.no_grad():                                                    # 不进行训练，禁用梯度
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)#计算训练集上损失
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)#计算验证集上损失
print("Training loss:", train_loss)                                      #（Training loss: 3.8258956909179687）
print("Validation loss:", val_loss)                                      #（Validation loss: 3.7619205951690673）


import time
start_time = time.time()                                                 # 模型开始训练时间
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1) #设置优化器
num_epochs = 2                                                           # 训练两轮
train_losses, val_losses, tokens_seen = train_model_simple(              # 调用之前的训练函数
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()                                                  # 结束时间
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")   #


from previous_chapters import plot_losses                               # 导入绘制损失函数图形
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))        # 横坐标
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)       # 绘制图形



"""
6.提取并保存回复
"""


torch.manual_seed(123)                                                  # 设置随机种子为123，确保生成的结果可复现
for entry in test_data[:3]:                                             # 遍历测试数据的前三个条目（假设`test_data`是一个包含测试样本的列表）

    input_text = format_input(entry)                                    # 格式化当前条目的输入文本（`entry`是测试数据中的一个样本）
    token_ids = generate(                                               # 使用模型生成token id
        model=model,                                                    # 使用的GPT模型
        idx=text_to_token_ids(input_text, tokenizer).to(device),        # 将输入文本转为token ids，并移动到指定的设备（如GPU）
        max_new_tokens=256,                                             # 最大生成256个新的tokens
        context_size=BASE_CONFIG["context_length"],                     # 上下文大小，使用配置中的`context_length`
        eos_id=50256                                                    # 结束标记的token id（通常是[EOS]标记）
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)            # 将生成的token ids转回文本
    response_text = (                                                   # 提取模型生成的文本部分
        generated_text[len(input_text):]                                # 去掉输入部分，仅保留生成部分
        .replace("### Response:", "")                                   # 去除生成文本中的"### Response:"标记
        .strip()                                                        # 去除生成文本两端的空格
    )

    print(input_text)                                                   # 打印输入文本
    print(f"\nCorrect response:\n>> {entry['output']}")                 # 打印正确的响应文本（从测试数据中获取）
    print(f"\nModel response:\n>> {response_text.strip()}")             # 打印模型生成的响应文本
    print("-------------------------------------")                      # 输出分隔线，用于区分每个样本的结果


from tqdm import tqdm                                                   # 导入tqdm库，用于显示进度条

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):       # 遍历`test_data`中的每个样本，并显示进度条
    input_text = format_input(entry)                                    # 格式化输入文本

    token_ids = generate(                                               # 使用模型生成token ids
        model=model,                                                    # 使用的GPT模型
        idx=text_to_token_ids(input_text, tokenizer).to(device),        # 将输入文本转换为token ids，并移动到指定的设备（如GPU）
        max_new_tokens=256,                                             # 生成最多256个新tokens
        context_size=BASE_CONFIG["context_length"],                     # 上下文大小，使用配置中的`context_length`
        eos_id=50256                                                    # 结束标记的token id
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)            # 将生成的token ids转回文本
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()  # 提取生成文本并去掉标记，去除两端的空白

    test_data[i]["model_response"] = response_text                      # 将模型生成的响应文本添加到`test_data`中的当前条目

# 将包含模型响应的测试数据保存到JSON文件中
with open("instruction-data-with-response.json", "w") as file:          # 打开文件以写入数据
    json.dump(test_data, file, indent=4)                                # 将`test_data`写入文件，并使用`indent=4`进行美化输出
print(test_data[0])                                                     #（key值有instruction，input，output和model_response）

import re                                                               # 导入re
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"            
torch.save(model.state_dict(), file_name)                               # 将模型参数也进行存储
print(f"Model saved as {file_name}")

# Load model via
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))


"""
7.评估微调后的模型

使用ollama在本地运行llama3模型来评估答案
"""

import psutil                                                           # 导入psutil库，用于获取和管理系统的进程信息

def check_if_running(process_name):                                     # 定义函数，用于检查进程是否正在运行
    running = False                                                     # 初始化标志位，默认进程没有运行
    for proc in psutil.process_iter(["name"]):                          # 遍历系统中的所有进程，获取进程名称
        if process_name in proc.info["name"]:                           # 如果进程名称包含给定的`process_name`
            running = True                                              # 说明进程正在运行
            break                                                       # 找到后退出循环
    return running                                                      # 返回进程是否运行的状态

ollama_running = check_if_running("ollama")                             # 检查"ollama"进程是否在运行

if not ollama_running:                                                  # 如果"ollama"进程没有运行
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")  # 抛出错误，提醒用户启动Ollama
print("Ollama running:", check_if_running("ollama"))                    # 打印Ollama是否正在运行的状态

import urllib.request                                                   # 导入urllib库，用于发送HTTP请求

def query_model(                                                        # 定义查询模型的函数
    prompt,                                                             # 输入的任务或问题
    model="llama3",                                                     # 使用的模型（默认是llama3）
    url="http://localhost:11434/api/chat"                               # 模型服务的URL地址
):
    # 创建数据负载（请求数据），包含模型名称、消息、以及一些选项设置
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}                         # 用户发送的内容
        ],
        "options": {                                                    # 设置模型响应的选项
            "seed": 123,                                                # 设置随机种子，确保生成的结果可复现
            "temperature": 0,                                           # 设置温度为0，确保生成的响应更加确定性
            "num_ctx": 2048                                             # 设置上下文的大小为2048
        }
    }

    # 将数据字典转为JSON格式字符串并编码为字节流
    payload = json.dumps(data).encode("utf-8")

    # 创建POST请求对象，并设置相关请求头
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")              # 设置请求头为JSON格式

    # 发送请求并获取响应
    response_data = ""                                                  # 初始化响应数据为空字符串
    with urllib.request.urlopen(request) as response:                   # 发送请求并接收响应
        # 持续读取响应数据直到结束
        while True:
            line = response.readline().decode("utf-8")                  # 读取一行数据并解码为UTF-8格式
            if not line:                                                # 如果没有数据则跳出循环
                break
            response_json = json.loads(line)                            # 将读取到的JSON格式字符串转换为字典
            response_data += response_json["message"]["content"]        # 将模型的回复内容追加到`response_data`中

    return response_data                                                # 返回模型的响应文本



model = "llama3"                                                        # 设置模型名称为"llama3"
result = query_model("What do Llamas eat?", model)                      # 查询模型关于“Llamas吃什么”的问题
print(result)                                                           # 打印模型的响应

for entry in test_data[:3]:                                             # 遍历测试数据的前三个条目
    prompt = (                                                          # 创建评分任务的提示文本
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response:")                                        # 打印数据集中的正确响应
    print(">>", entry['output'])
    print("\nModel response:")                                          # 打印模型生成的响应
    print(">>", entry["model_response"])
    print("\nScore:")                                                   # 打印评分
    print(">>", query_model(prompt))                                    # 查询模型并打印评分
    print("\n-------------------------")                                # 输出分隔线



def generate_model_scores(json_data, json_key, model="llama3"):         # 定义函数生成模型评分
    scores = []                                                         # 初始化评分列表
    for entry in tqdm(json_data, desc="Scoring entries"):               # 遍历JSON数据并显示进度条
        prompt = (                                                      # 创建评分任务的提示文本
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)                              # 查询模型评分
        try:
            scores.append(int(score))                                   # 尝试将评分转换为整数并添加到评分列表中
        except ValueError:                                              # 如果评分无法转换为整数，则打印错误信息
            print(f"Could not convert score: {score}")
            continue                                                    # 跳过当前条目

    return scores                                                       # 返回评分列表

scores = generate_model_scores(test_data, "model_response")             # 生成模型评分
print(f"Number of scores: {len(scores)} of {len(test_data)}")           # 打印生成的评分数量
print(f"Average score: {sum(scores)/len(scores):.2f}\n")                # 打印平均评分


