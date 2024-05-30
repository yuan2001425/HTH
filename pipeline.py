import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
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
model.generation_config.max_new_tokens = 1024

template = get_template(template_type, tokenizer)
seed_everything(42)

import difflib


def read_file_to_dict(filename):
    result_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            # 去除行末的换行符
            line = line.strip()
            # 查找左括号和右括号的位置
            left_paren_index = line.find('（')
            right_paren_index = line.find('）')
            if left_paren_index != -1 and right_paren_index != -1 and left_paren_index < right_paren_index:
                # 提取键和值
                key = line[left_paren_index + 1:right_paren_index]
                value = line[right_paren_index + 1:]
                # 去除值前后的空白字符
                value = value.strip()
                result_dict[key] = value
    return result_dict


# 示例文件名
filename = 'rag_database2.txt'

# 读取文件并转换为字典
my_dict = read_file_to_dict(filename)


def find_closest_key_value(dictionary, target_string, cutoff=0.6):
    # 获取与目标字符串最相似的键
    closest_keys = difflib.get_close_matches(target_string, dictionary.keys(), n=1, cutoff=cutoff)
    if closest_keys:
        # 返回最相似键对应的值
        closest_key = closest_keys[0]
        # return target_string, closest_key, dictionary[closest_key]
        return "（{}）{}".format(closest_key, dictionary[closest_key])
    else:
        return None


model_dir = '/media/oem/12T/HZY/HTH/deepseek-vl-7b-chat/v4-20240528-154547/checkpoint-380'
model = Swift.from_pretrained(model, model_dir, inference_mode=True)

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
q2_template = """
【任务背景】
我们让被试者画了一幅简笔画，下面是对这幅图像的【描述话语】与可能的心理推断

【描述话语】
-{}

【任务描述】
请你根据这些对图像的描述与可能的心理推断综合，使用连贯的自然语言连接起来

【开始任务】
如果准备好了请综合分析【描述话语】，并尝试描述被试者的心理状态，生成一段表达连贯的心理报告（中间不能换行）。要求不能添加额外的信息，缺少【描述话语】里面的信息。
"""

image_path = "/media/oem/12T/HZY/HTH/test/house.png"

# 【任务描述】
# 请你根据这些对图像的描述与可能的心理推断，综合判断被试者的心理状态，并尝试描述使用连贯的自然语言连接起来
# 【开始任务】
# 如果准备好了请综合分析【描述话语】，并尝试描述被试者的心理状态，生成一段表达连贯的报告。
def inf(model_, template_, query_):
    # res, _ = inference(model_, template_, query_)
    res = ""
    gen = inference_stream(model_, template_, query_, [])
    print_idx = 0
    for response, _ in gen:
        delta = response[print_idx:]
        print(delta, end='', flush=True)
        res += delta
        print_idx = len(response)
    # print("-----------\n", res)
    return res


query1 = q1_template.format(image_path)
print("【【Q】】:\n", query1, "\n【【A】】:\n")
response1 = inf(model, template, query1)
print("-------------------------")
print("-------------------------\n")

model_dir = '/media/oem/12T/HZY/HTH/deepseek-vl-7b-chat/v1-20240518-145315/checkpoint-380'
model = Swift.from_pretrained(model, model_dir, inference_mode=True)

arr = []
sentences = response1.split("；")
for sentence in sentences:
    if sentence:
        # 查找最相似的键并输出相应的值
        knowledge = find_closest_key_value(my_dict, sentence)
        arr.append(knowledge)
print("\n-------------------------")
print("-------------------------\n")
arr = set(arr)
query2 = q2_template.format("\n-".join(arr))
print("【【Q】】:\n", query2, "\n【【A】】:\n")
response2 = inf(model, template, query2)
