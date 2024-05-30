import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd

# 打开文件并读取JSON数据
with open('labels/legend_label2.json', 'r', encoding='utf-8') as file:
    json_data = file.read()

# 加载JSON数据
house, tree, human = json.loads(json_data)

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
model.generation_config.max_new_tokens = 1024

template = get_template(template_type, tokenizer)
seed_everything(42)

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
<img>/media/oem/12T/HZY/HTH/test/data/legend/house/{}</img>

【开始任务】
请在准备好后开始描述这幅简笔画。
"""


def find_columns_with_value_one(csv_file_path, large_img_num):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    # 遍历每一行
    arr = []
    for index, row in df.iterrows():
        if large_img_num in row[0]:
            # 找出值为1的列
            columns_with_one = row.index[row == 1].tolist()
            for i in columns_with_one:
                if "house" in csv_file_path:
                    arr.append(house[int(i) - 1])
                if "tree" in csv_file_path:
                    arr.append(tree[int(i) - 1])
                if "human" in csv_file_path:
                    arr.append(human[int(i) - 1])
            break
    return arr


large_img_folder = "test/data/legend/house/"
large_imgs = os.listdir(large_img_folder)

for large_img_path in large_imgs:
    ground_truth = find_columns_with_value_one('test/label_house.csv', large_img_path)
    if ground_truth:
        query1 = q1_template.format(large_img_path)
        print("【【Q】】:\n", query1, "\n【【A】】:\n")
        response, _ = inference(model, template, query1)
        response = response.split("；")
        print("图像路径：", large_img_path)
        print("真实标签：", ground_truth)
        print("预测标签：", response)
        correct_count = 0
        for sub in response:
            # 判断子串信息是否包含其中
            if any(sub in item for item in ground_truth):
                correct_count += 1
        print('真实标签个数: ', len(ground_truth))
        print('预测标签个数: ', len(response))
        accuracy = correct_count / len(ground_truth) if response else 0
        print(f"准确率: {accuracy * 100:.2f}%")
