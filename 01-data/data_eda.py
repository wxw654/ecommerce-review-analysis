# 1  导包
import pandas as pd
from collections import Counter  # 用来统计标签 {类别标签: 出现频率}
from config import Config
import matplotlib.pyplot as plt

# 2  创建配置对象，获取数据集路径
conf = Config()
file_path = conf.train_path

# 3  读取数据
data = pd.read_csv(
    file_path,  # 文件路径
    sep="\t",  # 指定分隔符
    names=["text", "label"]
)
print(f"数据总量:\t{len(data)}")
print(f"数据集前5行:\n{data.head()}")

# 4  统计标签分布 Counter
label_counts = Counter(data["label"])  # 统计各标签出现的次数
print(label_counts)  # label_counts 是一个 dict {类别标签: 出现频率}

print("标签分布")
for label, count in label_counts.items():
    print(f"标签 {label}:\t{count}次")

# 5  计算各个类别标签的 样本数量占比
total_rows = len(data)  # 总的样本数量
print("标签占比")
for label, count in label_counts.items():
    percentage = count / total_rows * 100  # 百分比
    print(f"标签 {label}:\t{percentage:.2f}%")

# 6  分析样本句子长度
# todo .str 访问字符串本身；.len() 返回每个字符串的长度
data["text_len"] = data["text"].str.len()
print(data.head())
print(f"平均长度为：{data['text_len'].mean():.2f} 字符")
print(f"最大长度为：{data['text_len'].max()} 字符")  # 38
print(f"最小长度为：{data['text_len'].min()} 字符")
print(f"长度标准差为：{data['text_len'].std():.2f}")

# 绘图展示
data["text_len"].plot(
    kind="hist",  # 直方图
    bins=20,  # 指定 20 个区间
)
plt.show()

# todo 数据比较整齐，没有出现样本类别不均衡的问题，确定最大长度 38
