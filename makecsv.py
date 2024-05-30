import csv
import json
import os

import pandas as pd
import random

# 打开文件并读取JSON数据
with open('labels/legend_label2.json', 'r', encoding='utf-8') as file:
    json_data = file.read()

# 加载JSON数据
house, tree, human = json.loads(json_data)


def find_columns_with_value_one(csv_file_path, wri):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    # 遍历每一行
    for index, row in df.iterrows():
        arr = []
        # 找出值为1的列
        columns_with_one = row.index[row == 1].tolist()
        for i in columns_with_one:
            if "house" in csv_file_path:
                arr.append(house[int(i) - 1])
            if "tree" in csv_file_path:
                arr.append(tree[int(i) - 1])
            if "human" in csv_file_path:
                arr.append(human[int(i) - 1])
        random.shuffle(arr)
        wri.writerow((row[0], "；".join(arr)))


output_file = 'labels/output5.csv'

# 写入CSV文件
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    find_columns_with_value_one('labels/label_house.csv', writer)
    find_columns_with_value_one('labels/label_tree.csv', writer)
    find_columns_with_value_one('labels/label_human.csv', writer)

large_img_folder = "data/picture/"
large_imgs = os.listdir(large_img_folder)


def find_columns_with_value_one_2(csv_file_path, large_img_num):
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


# 写入CSV文件
with open(output_file, 'a+', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for large_img in large_imgs:
        ground_truth = find_columns_with_value_one_2('labels/label_house.csv',
                                                     large_img) + find_columns_with_value_one_2('labels/label_tree.csv',
                                                                                                large_img) + find_columns_with_value_one_2(
            'labels/label_human.csv', large_img)
        random.shuffle(ground_truth)
        if ground_truth:
            writer.writerow((large_img_folder + large_img, "；".join(ground_truth)))
