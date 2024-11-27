"""
微调一个文本分类的模型

不同微调的分类：分类微调和指令微调
"""

#相关依赖的检测
from importlib.metadata import version                     
pkgs = ["matplotlib",
        "numpy",
        "tiktoken",
        "torch",
        "tensorflow", # For OpenAI's pretrained weights
        "pandas"      # Dataset loading
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

# 工具：防止某些代码单元格被重复执行

from IPython.core.magic import register_line_cell_magic                                                    # 从IPython导入自定义命令装饰器

executed_cells = set()                                                                                     # 创建一个集合用于存储已执行的单元格标识

@register_line_cell_magic                                                                                  # 注册一个新的IPython魔法命令
def run_once(line, cell):                                                                                  # 定义魔法命令的功能
    if line not in executed_cells:                                                                         # 如果当前单元格标识不在已执行的集合中
        get_ipython().run_cell(cell)                                                                       # 执行该单元格的内容
        executed_cells.add(line)                                                                           # 将单元格标识添加到集合中
    else:
        print(f"Cell '{line}' has already been executed.")                                                 # 如果已执行过，打印提示信息



"""
1.准备数据集

本实验要做的是区分是否为垃圾邮件
下载数据集->数据预处理（平衡数据集，label映射为数字、划分好训练集、验证集和测试集）
"""


# Step 1 :下载并解压数据集
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"                               # 下载地址
zip_path = "sms_spam_collection.zip"                                                                        # 压缩路径
extracted_path = "sms_spam_collection"                                                                      #
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():                                                                             # 判断是否下载过了
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:                                                              # Downloading the file
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:                                                         # Unzipping the file
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"                                         # Add .tsv file extension
    os.rename(original_file_path, data_file_path)                                                           # 把解压后的tsv文件重命名
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)                                 # 下载并解压


import pandas as pd
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])                            # 查看数据集中的数据
df
print(df["Label"].value_counts())                                                                           # 统计不同标签的数量（ham数量远远多于spam）

%%run_once balance_df
def create_balanced_dataset(df):                                                                            # 使数据集的不同类别均衡
    num_spam = df[df["Label"] == "spam"].shape[0]                                                           # Count the instances of "spam"
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)                                # Randomly sample "ham" instances to match the number of "spam" instances
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])                                        # Combine ham "subset" with "spam"
    return balanced_df

balanced_df = create_balanced_dataset(df)                           
print(balanced_df["Label"].value_counts())                                                                  #平衡后的数据集把多的类减少到和少的类一样多
%%run_once label_mapping
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})                                      # 把ham和spam分别映射到0 1


def random_split(df, train_frac, validation_frac):                                                          # 划分为训练集、验证集和测试集
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)                                         # 随机打乱整个DataFrame
    # 计算各部分的结束索引
    train_end = int(len(df) * train_frac)                                                                   # 训练集的结束索引
    validation_end = train_end + int(len(df) * validation_frac)                                             # 验证集的结束索引

    # 划分DataFrame
    train_df = df[:train_end]                                                                               # 切片获取训练集
    validation_df = df[train_end:validation_end]                                                            # 切片获取验证集
    test_df = df[validation_end:]                                                                           # 切片获取测试集

    return train_df, validation_df, test_df                                                                 # 返回三个DataFrame


train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)                                      # 调用函数对数据集进行划分，训练集占70%，验证集占10%，剩余20%为测试集
train_df.to_csv("train.csv", index=None)                                                                    # 保存为CSV文件
validation_df.to_csv("validation.csv", index=None)  
test_df.to_csv("test.csv", index=None)  



"""
2.创建数据加载器

把数据集转化为分批次的可迭代对象
统一文本长度（取最长或最短）
"""
import tiktoken                                                                                             # 导入依赖库
tokenizer = tiktoken.get_encoding("gpt2")                                                                   # 获取编码器
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))                                 # 让编码器允许特殊字符


import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):                                                                                # 定义一个继承自Dataset的SpamDataset类
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):                          # 初始化函数，接收csv文件路径、tokenizer、最大长度、pad_token_id
        self.data = pd.read_csv(csv_file)                                                                  # 读取CSV文件并存储为DataFrame

        # 预处理文本，进行tokenization（将文本转换为token）
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]                                           # 对DataFrame中的每个文本进行编码
        ]

        if max_length is None:                                                                             # 如果没有提供最大长度
            self.max_length = self._longest_encoded_length()                                               # 则计算编码文本中最长的长度作为最大长度
        else:
            self.max_length = max_length                                                                   # 否则使用提供的最大长度
            # 如果某些序列超过最大长度，则截断序列
            self.encoded_texts = [
                encoded_text[:self.max_length]                                                             # 截取最大长度
                for encoded_text in self.encoded_texts
            ]

        # 将所有序列填充到最大长度
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))                          # 用pad_token_id填充不足的部分
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):                                                                          # 获取指定索引的数据项
        encoded = self.encoded_texts[index]                                                                # 获取对应的编码文本
        label = self.data.iloc[index]["Label"]                                                             # 获取对应的标签（目标值）
        return (
            torch.tensor(encoded, dtype=torch.long),                                                       # 将编码文本转换为张量
            torch.tensor(label, dtype=torch.long)                                                          # 将标签转换为张量
        )

    def __len__(self):                                                                                     # 返回数据集的大小
        return len(self.data)

    def _longest_encoded_length(self):                                                                     # 计算最长的编码文本长度
        max_length = 0
        for encoded_text in self.encoded_texts:                                                            # 遍历所有编码后的文本
            encoded_length = len(encoded_text)                                                             # 获取编码文本的长度
            if encoded_length > max_length:                                                                # 如果当前文本长度大于最大长度
                max_length = encoded_length                                                                # 更新最大长度
        return max_length                                                                                  # 返回最大的文本长度

# 创建SpamDataset对象，传入训练数据路径、tokenizer和最大长度
train_dataset = SpamDataset(
    csv_file="train.csv",                                                                                  # 训练数据文件路径
    max_length=None,                                                                                       # 未指定最大长度
    tokenizer=tokenizer                                                                                    # 传入tokenizer
)

print(train_dataset.max_length)                                                                            # 打印数据集中最长文本的长度


val_dataset = SpamDataset(                                                                                 # 对验证集也采用相同的统一长度
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(                                                                                # 对测试集也采用一样的长度
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)




from torch.utils.data import DataLoader                                                                    # 导入数据加载器（将数据集转化为可迭代对象）

num_workers = 0                                                                                            # 子进程的数量设置为0
batch_size = 8                                                                                             # 批次大小为8

torch.manual_seed(123)                                                                                     # 设置随机数种子

train_loader = DataLoader(                                                                                 # 对训练集创建数据加载器
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(                                                                                   # 对验证集创建数据加载器
    dataset=val_dataset,    
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(                                                                                  # 对测试集创建数据加载器
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

print("Train loader:")                                                                                    
for input_batch, target_batch in train_loader:                                                             # 迭代数据加载器中最后一个批次
    pass        
print("Input batch dimensions:", input_batch.shape)                                                        #（Input batch dimensions: torch.Size([8, 120])）
print("Label batch dimensions", target_batch.shape)                                                        # (Label batch dimensions torch.Size([8]))

print(f"{len(train_loader)} training batches")                                                             # (130 training batches)
print(f"{len(val_loader)} validation batches")                                                             # (19 validation batches)
print(f"{len(test_loader)} test batches")                                                                  # (38 test batches)



"""
3.使用预训练的权重初始化模型
"""

CHOOSE_MODEL = "gpt2-small (124M)"                                                                         # 选择的GPT2模型版本为 "gpt2-small"，参数规模为124M
INPUT_PROMPT = "Every effort moves"                                                                        # 输入的初始提示文本

BASE_CONFIG = {
    "vocab_size": 50257,                                                                                   # 模型的词汇表大小
    "context_length": 1024,                                                                                # 最大上下文长度（模型输入序列的最大长度）
    "drop_rate": 0.0,                                                                                      # Dropout概率，0表示不使用dropout
    "qkv_bias": True                                                                                       # 是否在注意力机制中使用查询、键、值的偏置项
}

# 各种GPT2模型对应的特定参数配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},                                  # 小模型：嵌入维度768，12层，12个注意力头
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},                                # 中模型：嵌入维度1024，24层，16个注意力头
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},                                 # 大模型：嵌入维度1280，36层，20个注意力头
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},                                   # 超大模型：嵌入维度1600，48层，25个注意力头
}

# 将选择的模型的特定参数合并到基础配置中
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])                                                            # 根据选择的模型版本，更新基础配置

# 确保数据集的最大长度不超过模型的上下文长度
assert train_dataset.max_length <= BASE_CONFIG["context_length"], (                                        # 如果超出上下文长度则报错
    f"Dataset length {train_dataset.max_length} exceeds model's context " 
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "  
    f"`max_length={BASE_CONFIG['context_length']}`"
)



from gpt_download import download_and_load_gpt2                                                           # 导入用于下载并加载GPT2模型的方法
from previous_chapters import GPTModel, load_weights_into_gpt                                             # 导入GPT模型类和权重加载函数
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")                                          # 从选择的模型中提取模型大小（如"124M"）

# 下载并加载GPT2模型的配置和权重
settings, params = download_and_load_gpt2(
    model_size=model_size,                                                                                # 传入模型大小作为参数（如"124M"）
    models_dir="gpt2"                                                                                     # 指定存储GPT2模型的目录为 "gpt2"
)
model = GPTModel(BASE_CONFIG)                                                                             # 根据配置创建模型对象
load_weights_into_gpt(model, params)                                                                      # 加载预训练的GPT2权重到自定义模型中
model.eval()                                                                                              # 将模型设置为评估模式，禁用dropout等训练相关的功能

from previous_chapters import (                                                                           # 从之前的章节中导入所需的方法
    generate_text_simple,                                                                                 # 用于生成文本的简单方法
    text_to_token_ids,                                                                                    # 将文本转换为token ID的函数
    token_ids_to_text                                                                                     # 将token ID还原为文本的函数
)

text_1 = "Every effort moves you"                                                                         # 输入的提示文本
token_ids = generate_text_simple(                                                                         # 使用简单的生成文本方法
    model=model,                                                                                          # 指定使用的模型
    idx=text_to_token_ids(text_1, tokenizer),                                                             # 将提示文本转换为token ID
    max_new_tokens=15,                                                                                    # 生成的最大新token数为15
    context_size=BASE_CONFIG["context_length"]                                                            # 上下文长度设置为模型的最大长度
)
print(token_ids_to_text(token_ids, tokenizer))                                                            # 将生成的token ID还原为文本并打印输出(Every effort moves you forward.The first step is to understand the importance of your work)
# 从这里可以看出导入预训练的权重后，模型已经能够生成一个语意连贯的句子

# 文本2：判断是否为垃圾文本的提示
text_2 = (                                                                                                # 提供的输入文本，提示模型回答文本是否为垃圾内容
    "Is the following text 'spam'? Answer with 'yes' or 'no':"                                            # 提问提示
    " 'You are a winner you have been specially"                                                          # 垃圾内容示例部分
    " selected to receive $1000 cash or a $2000 award.'"                                                  # 垃圾内容示例部分
)
token_ids = generate_text_simple(  
    model=model,                   
    idx=text_to_token_ids(text_2, tokenizer),  
    max_new_tokens=23,              
    context_size=BASE_CONFIG["context_length"]  
)
print(token_ids_to_text(token_ids, tokenizer))                                                            # 在微调之前的模型是不能够实现分类功能的



"""
4.添加一个分类头

原来预训练最后一步映射到字典集上，这里换成映射到两个元素
"""

print(model)                                                                                             # 打印模型的架构信息
for param in model.parameters():                                                                         # 冻结原始模型
    param.requires_grad = False                                                                          # 把梯度设置为false，则不可训练

torch.manual_seed(123)                                                                                   # 设置随机数种子
num_classes = 2                                                                                          # 映射到两个元素
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)           # 修改线性层输入为嵌入层的输出，输出为2

# 理论上只需要对最后一层进行训练就行了
# 但是实验表明对一些额外的层也进行微调，能使模型的表现更好
for param in model.trf_blocks[-1].parameters():                                                          # 最后一个Transform块
    param.requires_grad = True
for param in model.final_norm.parameters():                                                              # 最后一个标准化层
    param.requires_grad = True  

inputs = tokenizer.encode("Do you have time")                                                            # 输入文本
inputs = torch.tensor(inputs).unsqueeze(0)                                                               # 展平
print("Inputs:", inputs)                                                                                 # （Inputs: tensor([[5211,  345,  423,  640]])）
print("Inputs dimensions:", inputs.shape)                                                                # （Inputs dimensions: torch.Size([1, 4])）

with torch.no_grad():                                                                                    # 不对模型进行训练
    outputs = model(inputs)                                                                              # 获取输出
print("Outputs:\n", outputs)                                                                             
print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)               #（Outputs dimensions: torch.Size([1, 4, 2])）
print("Last output token:", outputs[:, -1, :])                                                           # 最后一个token(Last output token: tensor([[-3.5983,  3.9902]]))



"""
5. 计算分类的损失和准确性
"""
print("Last output token:", outputs[:, -1, :])                                                           # 最后一个token(Last output token: tensor([[-3.5983,  3.9902]]))
probas = torch.softmax(outputs[:, -1, :], dim=-1)                                                        # 通过softmax进行归一化
label = torch.argmax(probas)                                                                             # 选取概率最大的标签
print("Class label:", label.item())                                                                      # （Class label: 1）

logits = outputs[:, -1, :]
label = torch.argmax(logits)                                                                             # 也可以直接取最大值，而不通过softmax归一化
print("Class label:", label.item())                                                                      # 结果不变 (Class label: 1）



def calc_accuracy_loader(data_loader, model, device, num_batches=None):                                 # 定义函数，计算模型在数据加载器上的准确率
    model.eval()                                                                                        # 将模型设置为评估模式，禁用dropout等训练特性
    correct_predictions, num_examples = 0, 0                                                            # 初始化正确预测数和样本总数为0

    if num_batches is None:                                                                             # 如果未指定批次数
        num_batches = len(data_loader)                                                                  # 默认使用整个数据加载器中的批次数
    else:
        num_batches = min(num_batches, len(data_loader))                                                # 使用指定批次数，但不能超过数据加载器的总批次数

    for i, (input_batch, target_batch) in enumerate(data_loader):                                       # 遍历数据加载器中的批次数据
        if i < num_batches:                                                                             # 仅处理指定数量的批次
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)                 # 将数据迁移到指定设备

            with torch.no_grad():                                                                       # 禁用梯度计算，节省内存和加速推理
                logits = model(input_batch)[:, -1, :]                                                   # 获取最后一个输出token的logits
            predicted_labels = torch.argmax(logits, dim=-1)                                             # 获取预测标签

            num_examples += predicted_labels.shape[0]                                                   # 累计样本总数
            correct_predictions += (predicted_labels == target_batch).sum().item()                      # 累计正确预测数（预测值等于目标值）
        else:
            break                                                                                       # 如果达到指定批次数，则退出循环
    return correct_predictions / num_examples                                                           # 返回准确率（正确预测数/样本总数）


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                   # 根据硬件环境选择运行设备（GPU或CPU）
model.to(device)                                                                                        # 将模型迁移到指定设备，无需赋值操作（对于 `nn.Module` 类）
torch.manual_seed(123)                                                                                  # 设置随机种子，确保因训练数据加载器随机打乱导致的结果可复现

train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)                      #限定为10个批次
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)  
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)  

# 打印训练集、验证集和测试集的准确率
print(f"Training accuracy: {train_accuracy*100:.2f}%")                                                  # （Training accuracy: 46.25%）
print(f"Validation accuracy: {val_accuracy*100:.2f}%")                                                  # （Validation accuracy: 45.00%）
print(f"Test accuracy: {test_accuracy*100:.2f}%")                                                       # （Test accuracy: 48.75%）


 
def calc_loss_batch(input_batch, target_batch, model, device):                                          # 定义函数，计算单个批次的损失
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)                         # 将输入和目标迁移到指定设备
    logits = model(input_batch)[:, -1, :]                                                               # 获取最后一个输出token的logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)                                      # 计算交叉熵损失
    return loss                                                                                         # 返回损失值

# 与第5章中的方法类似
def calc_loss_loader(data_loader, model, device, num_batches=None):                                     # 定义函数，计算数据加载器中多个批次的平均损失
    total_loss = 0.                                                                                     # 初始化总损失为0
    if len(data_loader) == 0:                                                                           # 如果数据加载器为空
        return float("nan")                                                                             # 返回NaN
    elif num_batches is None:                                                                           # 如果未指定批次数
        num_batches = len(data_loader)                                                                  # 默认使用数据加载器中的所有批次
    else:
                                                                                                        # 如果指定的批次数超过数据加载器中的总批次数，则限制为数据加载器的批次数
        num_batches = min(num_batches, len(data_loader))  
    for i, (input_batch, target_batch) in enumerate(data_loader):                                       # 遍历数据加载器中的批次
        if i < num_batches:                                                                             # 仅处理指定数量的批次
            loss = calc_loss_batch(input_batch, target_batch, model, device)                            # 计算当前批次的损失
            total_loss += loss.item()                                                                   # 累加损失值
        else:
            break                                                                                       # 如果达到指定批次数，则退出循环
    return total_loss / num_batches                                                                     # 返回平均损失（总损失/批次数）

with torch.no_grad():                                                                                   # 禁用梯度跟踪以提高效率，因为当前不是在训练模型
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)                           # 计算训练集的平均损失，限制为5个批次
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)                               # 计算验证集的平均损失，限制为5个批次
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)                             # 计算测试集的平均损失，限制为5个批次

# 打印训练集、验证集和测试集的平均损失，保留三位小数
print(f"Training loss: {train_loss:.3f}")                                                               # （Training loss: 2.453）
print(f"Validation loss: {val_loss:.3f}")                                                               # （Validation loss: 2.583）
print(f"Test loss: {test_loss:.3f}")                                                                    # （Test loss: 2.322）



"""
6.使用人工处理的数据对模型进行微调

和之前的模型训练方法基本相同
"""
# 定义训练分类器的函数
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # 初始化列表以记录损失和准确率
    train_losses, val_losses, train_accs, val_accs = [], [], [], []                                    # 用于保存训练和验证集的损失和准确率
    examples_seen, global_step = 0, -1                                                                 # 记录处理的样本数和全局步数

    # 训练主循环
    for epoch in range(num_epochs):                                                                   # 遍历每个epoch
        model.train()                                                                                 # 将模型设置为训练模式

        for input_batch, target_batch in train_loader:                                                # 遍历训练数据加载器中的每个批次
            optimizer.zero_grad()                                                                     # 清除上一批次计算的梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)                          # 计算当前批次的损失
            loss.backward()                                                                           # 反向传播计算梯度
            optimizer.step()                                                                          # 使用梯度更新模型权重
            examples_seen += input_batch.shape[0]                                                     # 累加处理的样本数
            global_step += 1                                                                          # 增加全局步数

            # 可选的评估步骤
            if global_step % eval_freq == 0:                                                          # 每经过eval_freq个步骤评估一次
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)                               # 评估训练集和验证集的损失
                train_losses.append(train_loss)                                                       # 保存训练集损失
                val_losses.append(val_loss)                                                           # 保存验证集损失
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")                        # 打印当前epoch和step的损失

        # 每个epoch结束时计算训练集和验证集的准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")                            # 打印训练集准确率
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")                                       # 打印验证集准确率
        train_accs.append(train_accuracy)                                                            # 保存训练集准确率
        val_accs.append(val_accuracy)                                                                # 保存验证集准确率

    return train_losses, val_losses, train_accs, val_accs, examples_seen                             # 返回损失和准确率的列表以及样本数

# 定义评估模型的函数
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()                                                                                     # 将模型设置为评估模式
    with torch.no_grad():                                                                            # 禁用梯度计算以节省内存
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)            # 计算训练集损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)                # 计算验证集损失
    model.train()                                                                                    # 将模型恢复为训练模式
    return train_loss, val_loss                                                                      # 返回训练集和验证集的损失


import time                                                                                          # 导入time模块以计算训练时间
start_time = time.time()                                                                             # 记录训练开始时间
torch.manual_seed(123)                                                                               # 设置随机种子以确保结果可复现
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)                         # 定义AdamW优化器
num_epochs = 5                                                                                       # 设置训练的总epoch数
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,                                                # 训练时每50步评估一次，每次评估5个批次
)
end_time = time.time()                                                                               # 记录训练结束时间
execution_time_minutes = (end_time - start_time) / 60                                                # 计算训练时间（分钟）
print(f"Training completed in {execution_time_minutes:.2f} minutes.")                                # 打印训练所需时间


import matplotlib.pyplot as plt                                                                      # 导入matplotlib用于绘制图表
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))                                                          # 创建绘图区域

    # 绘制训练和验证的损失或准确率曲线
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")                                   # 绘制训练集曲线
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")                   # 绘制验证集曲线
    ax1.set_xlabel("Epochs")                                                                         # 设置x轴标签
    ax1.set_ylabel(label.capitalize())                                                               # 设置y轴标签
    ax1.legend()                                                                                     # 显示图例

    # 创建一个新的x轴，用于显示处理的样本数量
    ax2 = ax1.twiny()                                                                                # 创建一个共享y轴的第二个x轴
    ax2.plot(examples_seen, train_values, alpha=0)                                                   # 隐藏的曲线，仅用于对齐坐标轴
    ax2.set_xlabel("Examples seen")                                                                  # 设置第二个x轴的标签

    fig.tight_layout()                                                                               # 调整布局，避免标签重叠
    plt.savefig(f"{label}-plot.pdf")                                                                 # 保存图表为PDF文件
    plt.show()                                                                                       # 显示图表


# 绘制损失变化图
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))                                     # 生成每个epoch的索引
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))                           # 生成每个epoch对应的样本数
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)                           # 绘制训练和验证损失的变化曲线

# 绘制准确率变化图
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))                                       # 生成每个epoch的索引
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))                             # 生成每个epoch对应的样本数
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")             # 绘制训练和验证准确率的变化曲线

# 计算并打印最终的训练、验证和测试准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device)                                  
val_accuracy = calc_accuracy_loader(val_loader, model, device)                                     
test_accuracy = calc_accuracy_loader(test_loader, model, device)                                 
print(f"Training accuracy: {train_accuracy*100:.2f}%")                                               # (Training accuracy: 97.21%)
print(f"Validation accuracy: {val_accuracy*100:.2f}%")                                               # (Validation accuracy: 97.32%)
print(f"Test accuracy: {test_accuracy*100:.2f}%")                                                    # （Test accuracy: 95.67%）


"""
7.使用微调之后的模型

模型能够较好地实现分类地功能
"""

# 定义文本分类函数
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()                                                                                    # 设置模型为评估模式

    # 准备输入给模型的数据
    input_ids = tokenizer.encode(text)                                                              # 使用tokenizer将文本编码为token IDs
    supported_context_length = model.pos_emb.weight.shape[0]                                        # 获取模型支持的最大上下文长度（位置编码的长度）
    # 如果输入文本太长，则截断
    input_ids = input_ids[:min(max_length, supported_context_length)]                               # 保证输入不超过模型支持的最大长度
    # 对序列进行填充，以确保长度一致
    input_ids += [pad_token_id] * (max_length - len(input_ids))                                     # 用pad_token_id填充直到达到最大长度
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)                              # 将输入转换为tensor并添加批次维度（unsqueeze(0)）

    # 模型推理
    with torch.no_grad():                                                                           # 禁用梯度计算，提高推理效率
        logits = model(input_tensor)[:, -1, :]                                                      # 获取模型最后一个token的logits（输出）
    predicted_label = torch.argmax(logits, dim=-1).item()                                           # 获取最大logits值对应的索引作为预测标签

    # 返回分类结果
    return "spam" if predicted_label == 1 else "not spam"                                           # 如果预测标签为1，则认为是"垃圾邮件"，否则为"非垃圾邮件"


text_1 = (                                                                                          # 第一段文本
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(classify_review(                                                                              # （spam）
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

text_2 = (                                                                                          # 第二段文本
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)   
print(classify_review(                                                                              #(not spam)
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

torch.save(model.state_dict(), "review_classifier.pth")                                             #保存微调后模型的权重
model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)      # 重新加载
model.load_state_dict(model_state_dict)