import pandas as pd

csv_file_path = "features.csv"

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 写入CSV文件
with open('rag_database.txt', 'w', newline='', encoding='utf-8') as file:
    for index, row in df.iterrows():
        text = "（" + row[2] + "）" + row[3]
        file.write(text + "\n")
        print(text + "\n---")
