import csv
import json

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
<img>/media/oem/12T/HZY/HTH/{}</img>

【开始任务】
请在准备好后开始描述这幅简笔画。
"""


def csv_to_json(file_path):
    conversations_list = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            conversation = {
                "conversations": [
                    {
                        "from": "user",
                        "value": q1_template.format(row[0])
                    },
                    {
                        "from": "assistant",
                        "value": row[1]
                    }
                ]
            }
            conversations_list.append(conversation)
    return json.dumps(conversations_list, indent=4, ensure_ascii=False)


# 使用示例
file_path = 'labels/output5.csv'  # 替换为你的CSV文件路径
json_output = csv_to_json(file_path)
print(json_output)

with open('dataset3.json', 'w') as file:
    # 将JSON字符串写入文件
    file.write(json_output)
