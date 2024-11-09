"""
比较不同BPE方法（对单个陌生token进行拆分）的效率
"""

"""
1.使用tiktoken开源中的BPE方法（利用算法对GPT-2原来使用的BPE进行了优化）
"""

from importlib.metadata import version                                           #导入依赖库
print("tiktoken version:", version("tiktoken"))                                  #检测

import tiktoken
tik_tokenizer = tiktoken.get_encoding("gpt2")                                    #设置编码器
text = "Hello, world. Is this-- a test?"
integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})         #将字符编码为ID
print(integers)                                                                  #（[15496, 11, 995, 13, 1148, 428, 438, 257, 1332, 30]）
strings = tik_tokenizer.decode(integers)                                         #再解码为字符串
print(strings)                                                                   #（Hello, world. Is this-- a test?）
print(tik_tokenizer.n_vocab)                                                     #显示供编码用到字典的字符总数（50257）


"""
2.使用GPT-2最初使用的BPE方法
"""

from bpe_openai_gpt2 import get_encoder, download_vocab                          #导入gptBPE方法 
download_vocab()                                                                 #下载词汇集
orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")            #设置编码解码器

integers = orig_tokenizer.encode(text)                                           #编码
print(integers)                                                                  #（[15496, 11, 995, 13, 1148, 428, 438, 257, 1332, 30]）
strings = orig_tokenizer.decode(integers)                                        #解码
print(strings)                                                                   #Hello, world. Is this-- a test?


"""
3.使用Hugging Face的transforms库中的BPE方法
"""

import transformers                                                             #下载依赖包
transformers.__version__                                                        #检测

from transformers import GPT2Tokenizer
hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")                            #初始化tokenizer，对应的模型为gpt2
hf_tokenizer(strings)["input_ids"]                                              #打印编码后的token IDs（[15496, 11, 995, 13, 1148, 428, 438, 257, 1332, 30]）
 

 
"""
4.benchmark
"""
with open('/content/the-verdict.txt', 'r', encoding='utf-8') as f:             #读取第一小节中的短片小说文本
    raw_text = f.read() 

#使用Jupyter Notebook中的%timeit指令，返回代码平均执行时间
%timeit orig_tokenizer.encode(raw_text)                                        #原始GPT2的方法（15.2 ms ± 2.89 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)）
%timeit tik_tokenizer.encode(raw_text)                                         #tiketoken库中的方法（2.44 ms ± 51.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)）
%timeit hf_tokenizer(raw_text)["input_ids"]                                    #transformer库中的方法（Token indices sequence length is longer than the specified maximum sequence length for this model (5145 > 1024). Running this sequence through the model will result in indexing errors31.4 ms ± 8.95 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)）
%timeit hf_tokenizer(raw_text, max_length=5145, truncation=True)["input_ids"]  #transformer库中的方法,指定最大长度，超过就截断（33.2 ms ± 9.81 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

#比较下来，似乎tiktoken库中的BPE方法最高效