"""
Input Embedding pipline 
"""


from importlib.metadata import version

print("torch version:", version("torch"))                             #检测依赖库是否成功安装
print("tiktoken version:", version("tiktoken"))                       #检测依赖库是否成功安装


"""
1.对短篇小说 The Verdict by Edith Wharton的内容拆分为一个个token
"""

import os
import urllib.request
if not os.path.exists("the-verdict.txt"):                               #若指定当前文件目录下不存在要求文本               
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)                          #从网络上下载文本存储在对应位置


with open("the-verdict.txt", "r", encoding="utf-8") as f:               #读取文本内容
    raw_text = f.read()
print("Total number of character:", len(raw_text))                      #打印文本长度（Total number of character: 20479）       
print(raw_text[:99])                                                    #打印前99个字符（I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no）

#正则表达式使用
import re                                                               #将文件根据正则表达式规则拆分
text = "Hello, world. This, is a test." 
result = re.split(r'(\s)', text)                                        #匹配文本中所有空格，将文本拆分
print(result)                                                           #（['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']）

result = re.split(r'([,.]|\s)', text)                                   #更换拆分规则，句号逗号或空格
print(result)                                                           #（['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is', ' ', 'a', ' ', 'test', '.', '']）
                                                                        #匹配到的元素作为单独存入列表
result = [item for item in result if item.strip()]                      #去除结果中的空串
print(result)                                                           #['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)                      #进一步加强对正则表达式的要求，对更多符号进行划分
result = [item.strip() for item in result if item.strip()]
print(result)                                                           #['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']


preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)            #对刚才短篇小说文本中数据拆分
preprocessed = [item.strip() for item in preprocessed if item.strip()]  #去空格
print(preprocessed[:30])                                                #打印前30个元素（['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to', 'hear', 'that', ',', 'in']）
print(len(preprocessed))                                                #（4690）


"""
2.将拆分后的token数据转换为tokenID
"""

all_words = sorted(set(preprocessed))                                   #对原来数据元素去重排序
vocab_size = len(all_words)                                             #去重后元素个数
print(vocab_size)                                                       #(1130)
vocab = {token:integer for integer,token in enumerate(all_words)}       #生成一个字典集合，将每个token与它的ID对应

for i, item in enumerate(vocab.items()):                                #打印该字典前五十个数据
    print(item)
    if i >= 50:
        break


class SimpleTokenizerV1:                                                #通过原有的字典匹配，将新文本也都用数字序列表示
    def __init__(self, vocab):                                          #初始化
        self.str_to_int = vocab                                         #字符到数字的字典集合
        self.int_to_str = {i:s for s,i in vocab.items()}                #数字到字符的字典集合
    
    def encode(self, text):                                             #编码
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)        #正则化
                                
        preprocessed = [                                                #去空格
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]                #字符列表转数字列表
        return ids
        
    def decode(self, ids):                                              #解码
        text = " ".join([self.int_to_str[i] for i in ids])              #数字列表转字符列表，再转字符串，用空格隔开
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)                 #将符号前的空格去除（用第一个字符（从0开始）代替匹配到的所有字符）
        return text          


tokenizer = SimpleTokenizerV1(vocab)                                    #初始化（传递了从短篇小说中获取的字典）
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)                                            #编码
print(ids)                                                              #（[1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]）
tokenizer.decode(ids)                                                   #再进行解码（'" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'）
tokenizer.decode(tokenizer.encode(text))                                #先编码再解码，可逆


"""
3.添加一些特殊标记来表示未知的单词和文本

最原始：起始[BOS]，结束[EOS]，padding[PAD]，未知[UNK]
GPT-2：结束<|endoftext|>，padding<|endoftext|>，未知->拆分为子token
实验：两个不同文本源中，用<|endoftext|>隔开
"""

okenizer = SimpleTokenizerV1(vocab)
text = "Hello, do you like tea. Is this-- a test?"
tokenizer.encode(text)
#以上代码会报错，因为Hello并不在我们的词汇表中

all_tokens = sorted(list(set(preprocessed)))                                   #原始的字符序列，排序
all_tokens.extend(["<|endoftext|>", "<|unk|>"])                                #拓展字符序列（加入特殊字符）
vocab = {token:integer for integer,token in enumerate(all_tokens)}             #创建对应字典
len(vocab.items())                                                             #（1132）比原来多2

for i, item in enumerate(list(vocab.items())[-5:]):                            #打印字典中的最后5个元素
    print(item)                                                                #('younger', 1127)('your', 1128)('yourself', 1129)('<|endoftext|>', 1130)('<|unk|>', 1131)
 

class SimpleTokenizerV2:                                                       #对原来的编码解码进行相应调整，以适应特殊标签  
    def __init__(self, vocab):                                                 #构造函数，和之前一样
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):                                                    #编码
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)               #字符序列
        preprocessed = [item.strip() for item in preprocessed if item.strip()] #去空
        preprocessed = [                                                          
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed                            #核心逻辑，若字符序列中没有，则视作<|unk|>
        ]

        ids = [self.str_to_int[s] for s in preprocessed]                       #转化为数字序列
        return ids
        
    def decode(self, ids):                                                     #和之前一样
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    

tokenizer = SimpleTokenizerV2(vocab)                                          #用新的标记器处理两段含未知token的文本源
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."       
text = " <|endoftext|> ".join((text1, text2))                                 #将不同文本源用<|endoftext|>连接
print(text)                                                                   #(Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.)
#编码与解码
tokenizer.encode(text)                                                        #([1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7])
tokenizer.decode(tokenizer.encode(text))                                      #(<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.)


"""
4.BytePair encoding(BPE):分词器编码:允许对未知单词进行拆分

实验中采用了tiktoken开源库里的分词器（利用优化算法比原始的GPT-2的BPR分词器效率高5倍）
"""

import importlib
import tiktoken
print("tiktoken version:", importlib.metadata.version("tiktoken"))            #确认tiktoken库成功导入
tokenizer = tiktoken.get_encoding("gpt2")                                     #初始化分词器（导入的gpt2，应该是对应字典集合）
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})          #编码,允许<|endoftext|>标签
print(integers)                                                               #（[15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13])
strings = tokenizer.decode(integers)                                          #解码
print(strings)                                                                #（还原）


"""
5.使用滑动窗口进行数据采样（用于训练模型的预测，每次预测一个单词），x为原来窗口中数据，y为窗口右移一位里面的数据
"""

with open("the-verdict.txt", "r", encoding="utf-8") as f:                     #打开短篇小说对应文本
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)                                         #对原始文本进行编码（数字序列化）
print(len(enc_text))                                                          #（5145）

enc_sample = enc_text[50:]                                                    #编码后的前50个元素
context_size = 4                                                              #将滑动窗口的大小设置为4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]                                              #窗口右移一位
print(f"x: {x}")                                                              #（x: [290, 4920, 2241, 287]）
print(f"y:      {y}")                                                         #（y:      [4920, 2241, 287, 257]）

for i in range(1, context_size+1):
    context = enc_sample[:i]                                                  #0->i-1的元素
    desired = enc_sample[i]                                                   #第i个元素
    print(context, "---->", desired)                                          #（[290] ----> 4920，[290, 4920] ----> 2241，[290, 4920, 2241] ----> 287，[290, 4920, 2241, 287] ----> 257）

for i in range(1, context_size+1):                                            #推测的同时解码
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))    #（ and ---->  established，and established ---->  himself，and established himself ---->  in，and established himself in ---->  a）


from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):                                                  #创建数据集，从输入文本数据集中提取块
    def __init__(self, txt, tokenizer, max_length, stride):                  
        self.input_ids = []                                                   #初始化x和y为空列表
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})  #将文本编码

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):               #stride是循环中的步长  
            input_chunk = token_ids[i:i + max_length]                         #一个窗口中的数据作为input
            target_chunk = token_ids[i + 1: i + max_length + 1]               #窗口右移一位数据作target
            #self.input_ids.append(torch.tensor(input_chunk))                 #列表转换为张量存储
            #self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):                                                        #返回input张量的长度
        return len(self.input_ids)

    def __getitem__(self, idx):                                               #放回第i个片段的input和target
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256,                   #创建数据加载器（max_length为窗口大小，stride！=max_length可让不同窗口中由重复值）
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")                                 #采用开源库tiktoken中的分词器
 
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)                #初始化数据集

    # Create dataloader
    dataloader = DataLoader(                                                  #初始化数据加载器
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader                                                         #返回创建的数据加载器


with open("the-verdict.txt", "r", encoding="utf-8") as f:                     #打开数据集
    raw_text = f.read()
dataloader = create_dataloader_v1(                                            #创建数据加载器
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False             #窗口大小为4，步长为1，批次大小为1（每次取一个样本）
)

data_iter = iter(dataloader)                                                  #将数据加载器 dataloader 转换为一个迭代器 data_iter
first_batch = next(data_iter)                                                 #获取数据迭代器中的第一个批次（一个input张量，一个target张量）
print(first_batch)                                                            #（[tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]）
second_batch = next(data_iter)                                                #获取第二个批次
print(second_batch)                                                           #（[tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]）

#过多的重叠可能导致过拟合
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)         #步幅和窗口一样大，无重叠
data_iter = iter(dataloader)                                                  #测试批次大小不为1的情况
inputs, targets = next(data_iter)                                             #获取inputs和targets
print("Inputs:\n", inputs)                                                    #一次输出八个x张量构成的张量
print("\nTargets:\n", targets)


"""
6.创建token嵌入（将所有的tokenID转换为一个连续的向量）
"""
import torch
input_ids = torch.tensor([2, 3, 5, 1])                                       #假设输入中只有4个token

vocab_size = 6                                                               #假设我们的词汇表只有 6 个单词
output_dim = 3                                                               #并且我们想要创建大小为 3 的嵌入
torch.manual_seed(123)                                                       #设置随机化种子，嵌入层权重初始化是随机的
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)                 #创建一个嵌入层（指定词汇表大小，和嵌入后的向量维度）
print(embedding_layer.weight)                                                #将指定词汇表的数据各自映射到一个指定维度的向量上（一个6*3的权重矩阵）

print(embedding_layer(torch.tensor([3])))                                    #(tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>))
print(embedding_layer(input_ids))                                            #（打印出经过嵌入层后的数字序列对应的张量集合）


"""
7.编码词的位置(不处理的话不同位置的同一个词，对应的嵌入向量是一样的)

input embedding=token embedding + position embedding
"""

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)           #toke的嵌入,50257个token，嵌入为256维度

max_length = 4
dataloader = create_dataloader_v1(                                           #窗口大小和步长设置为4，批次大小为8
    raw_text, batch_size=8, max_length=max_length                 ,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)                                                 #转换为可迭代对象
inputs, targets = next(data_iter)                                            #第一个批次数据

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)                                     #（torch.Size([8, 4])）（8个窗口数据，每个窗口中4个token）

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)                                                #（torch.Size([8, 4, 256]）（将每个token映射到了一个256维的张量）

#GPT-2采用的位置嵌入是绝对位置嵌入
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)         #再创建一个嵌入层，元素个数为一个窗口的大小，输出维度应该要和token嵌入的维度一样）

pos_embeddings = pos_embedding_layer(torch.arange(max_length))               #arange->生成0 到 max_length - 1 的一维整数序列张量
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings                         #共同构成最终的输入嵌入
print(input_embeddings.shape)                                                #torch.Size([8, 4])会广播为torch.Size([8, 4, 256]，从而对这批次中的八个input都嵌入



