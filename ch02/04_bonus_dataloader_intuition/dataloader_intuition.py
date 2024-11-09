"""
对数据加载（dataloader）的直观理解
"""

from importlib.metadata import version                                           #导入依赖库
import torch
print("torch version:", version("torch"))                                        #检测


with open("number-data.txt", "w", encoding="utf-8") as f:                        #写入文件
    for number in range(1001):                                                   #写入0到1000的数据，作为tokenlize后的文本数据
        f.write(f"{number} ")


from torch.utils.data import Dataset, DataLoader                                 #导入
class GPTDatasetV1(Dataset):                                                     #构建数据集（Dataset）,这里为了简化直接从文本中读取token_ids
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []                                                      #初始化input_ids
        self.target_ids = []                                                     #初始化target_ids

        # Modification
        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        token_ids = [int(i) for i in txt.strip().split()]                        #读取文本中的token_ids,去空转化为列表

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):                  #设置范围和步长
            input_chunk = token_ids[i:i + max_length]                            #input窗口中的数据
            target_chunk = token_ids[i + 1: i + max_length + 1]                  #target窗口中的数据
            self.input_ids.append(torch.tensor(input_chunk))                     #存入
            self.target_ids.append(torch.tensor(target_chunk))                   #存入

    def __len__(self):                                                           #获取input_ids的数目，即为批次数
        return len(self.input_ids)

    def __getitem__(self, idx):                                                  #索取input_id和对应target_id
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
                                                                                 #构建数据加载器(Dataloader)
    # Initialize the tokenizer
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = None                                                             #这里便不需要编码了

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)                   #初始化数据集

    # Create dataloader  
    dataloader = DataLoader(                                                     #创建数据加载对象
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader                                                            #返回数据加载器

with open("number-data.txt", "r", encoding="utf-8") as f:                        #读取数字文本
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)                                                     #将数据加载器转化为可迭代对象
first_batch = next(data_iter)                                                    #第一个批次数据
print(first_batch)                                                               #（[tensor([[0, 1, 2, 3]]), tensor([[1, 2, 3, 4]])]）

dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False)
for inputs, targets in dataloader:                                               #将批次大小设置为2，观察结果
    pass
print("Inputs:\n", inputs)                                                       #打印出的其实是最后一个批次的数据
print("\nTargets:\n", targets)

torch.manual_seed(123)                                                           #设置随机数种子，保证每次打乱的结果一样
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=True)
for inputs, targets in dataloader:                                               #遍历时批次顺序被打乱，即不一定从第一批开始，到第二批；
    pass
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)