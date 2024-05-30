import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# embedding model: m3e-base
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "moka-ai/m3e-base"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    cache_folder="/home/oem/.cache/huggingface",
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为文本生成向量表示用于文本检索",
)

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)
from swift.tuners import Swift
from swift.utils import seed_everything
import torch

model_type = ModelType.deepseek_vl_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
print("Model is on device:", model.device)
model.generation_config.max_new_tokens = 512

# model_dir = '/media/oem/12T/HZY/HTH/deepseek-vl-7b-chat/v1-20240518-145315/checkpoint-380'
# model_dir = '/media/oem/12T/HZY/HTH/deepseek-vl-7b-chat/v2-20240527-171228/checkpoint-380'
# model_dir = '/media/oem/12T/HZY/HTH/deepseek-vl-7b-chat/v3-20240527-174823/checkpoint-380'
model_dir = '/media/oem/12T/HZY/HTH/deepseek-vl-7b-chat/v4-20240528-154547/checkpoint-380'
model = Swift.from_pretrained(model, model_dir, inference_mode=True)

template = get_template(template_type, tokenizer)
seed_everything(42)

q1_template = """
【任务背景】
在心理评估领域，房树人（HTP）测试是一种常用的投射性绘画技术，用以探索个体的心理特征和内在感受。研究人员通过分析被试者所绘制的图画来获取相关信息。

【任务描述】
请根据提供的HTP测试简笔画图像，进行详细的观察，并从专业角度描述画面中的元素。给你的图片有可能是房屋、树木、人物或是他们的组合

【描述话语】
描述应简洁明了，避免冗余，尽量使用直观和易于理解的语言。也应专业、准确，并使用心理学专业术语，以便于学术讨论和研究。

【面向对象】
面向心理学研究人员、心理咨询师、临床心理学家和其他相关领域的专家学者。所以需要保持描述的客观性和科学性，避免主观臆断。

【图像信息】
<img>{}</img>

【开始任务】
请在准备好后开始描述这幅简笔画。
"""
# q2_template = """
# 【任务描述】
# 请你根据对于房树人图像的描述话语与可能用得到的心理映射知识判断被试者可能的心理状态。
# 请你尽可能找出所有能用到的知识并剔除不需要的知识后再进行分析。
#
# 【描述话语】
# {}
#
# 【知识库】
# 注意：给你的每条知识均符合格式$（A）B$，其中$A$是图中可能存在的相关特征，$B$代表着何种被试者的心理映射
#
# 以下是知识：
# -{}
#
# 【特别注意】
# 【知识库】中仅仅代表可能用到的信息，不一定需要全部都使用，请仔细判断甄别。与【描述话语】中无关的知识请不要使用。
#
# 【开始任务】
# 如果准备好了请分析【描述话语】中对应被试者的心理状态，输出一段话即可，不要超过50字。
# """
# q3_template = """
# 【任务描述】
# 请你根据这些对于被试者所画图像的描述与可能的心理推断综合判断被试者的心理状态并尝试描述使用自然语言描述
#
# 【描述话语】
# -{}
#
# 【开始任务】
# 如果准备好了请综合分析【描述话语】，并尝试描述被试者的心理状态。
# """

image_path = "/media/oem/12T/HZY/HTH/test/house.png"

query1 = q1_template.format(image_path)
print("【【Q】】:\n", query1)
# print(model.cuda())
response1, _ = inference(model, template, query1)
print("【【A】】:\n", response1)
print("-------------------------")
print("-------------------------\n")

from rag_test import rag_test

rag_test(response1, embedding)

#
# arr = []
# sentences = response1.split("；")
# for sentence in sentences:
#     if sentence:
#         query2 = q2_template.format(sentence, "\n-".join([a.page_content for a in db.similarity_search(sentence)]))
#         print("【【Q】】:\n", query2)
#         response2, _ = inference(model, template, query2)
#         arr.append(response2)
#         print("【【A】】:\n", response2)
#         print("\n*************************")
# print("\n-------------------------")
# print("-------------------------\n")
# query3 = q3_template.format("\n-".join(arr))
# print("【【Q】】:\n", query3)
# response3, _ = inference(model, template, query3)
# print("【【A】】:\n", response3)
