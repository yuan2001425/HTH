import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

loader = TextLoader("rag_database2.txt")
documents = loader.load()
# 创建拆分器
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=10,
    chunk_overlap=5
)
# 拆分文档
texts = text_splitter.split_documents(documents)

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)

from swift.utils import seed_everything
import torch

model_type = ModelType.deepseek_vl_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 512

template = get_template(template_type, tokenizer)
seed_everything(42)

q2_template = """
【任务描述】
请你根据这些对于被试者所画图像的描述与可能的心理推断综合判断被试者的心理状态并尝试描述使用自然语言描述

【描述话语】
-{}

【开始任务】
如果准备好了请综合分析【描述话语】，并尝试描述被试者的心理状态。
"""


def rag_test(response1, embedding):
    # 数据入库
    db = Chroma.from_documents(texts, embedding)
    arr = []
    sentences = response1.split("；")
    for sentence in sentences:
        if sentence:
            knowledge = db.similarity_search(sentence)[0].page_content
            arr.append(knowledge)
    print("\n-------------------------")
    print("-------------------------\n")
    query3 = q2_template.format("\n-".join(arr))
    print("【【Q】】:\n", query3)
    response3, _ = inference(model, template, query3)
    print("【【A】】:\n", response3)
