# 1  导包 读取配置信息
from config import Config
import os
import jieba

conf = Config()

# 2  获取原始数据路径
train_data_file = conf.train_path  # 原始训练集路径
test_data_file = conf.test_path  # 原始测试集路径
dev_data_file = conf.dev_path  # 原始验证集路径

# 3  验证: 检查三个数据文件是否存在
if (
        not os.path.exists(train_data_file)  # 如果文件不存在，则 True
        or not os.path.exists(test_data_file)
        or not os.path.exists(dev_data_file)
):
    print("文件不存在，请你检查文件路径")
    raise SystemExit  # 抛出异常 不存在

# 4  根据分词策略，确定分词方式
if conf.use_char_segmentation:
    mode = "char"  # 按照字符分词
else:
    mode = "jieba"  # 按照 jieba 分词

# 文本处理后的保存路径
preprocess_train_data_path = os.path.join(  # 拼接文件路径
    conf.final_data, f"train_ft_{mode}.txt"
)
preprocess_test_data_path = os.path.join(  # 拼接文件路径
    conf.final_data, f"test_ft_{mode}.txt"
)
preprocess_dev_data_path = os.path.join(  # 拼接文件路径
    conf.final_data, f"dev_ft_{mode}.txt"
)

# 5  读取类别文件
with open(conf.class_data_path, "r", encoding="utf-8") as f:
    content = f.read().strip()

label_list = content.split()
# ['finance', 'realty', 'stocks', 'education', 'science',
# 'society', 'politics', 'sports', 'game', 'entertainment']
print(label_list)

id2label = {idx: label for idx, label in enumerate(label_list)}
# {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science', 5:
# 'society', 6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}
print(id2label)


# 6  具体分词
def preprocess(data_file, save_path):
    '''
    :param data_file: 原始数据文件的路径
    :param save_path: 处理后的文件保存路径
    '''
    # 1  打开原始数据文件，准备读取
    count = 0
    with open(data_file, "r", encoding="utf-8") as fr:
        # 2  打开保存文件路径，准备写入
        with open(save_path, "w", encoding="utf-8") as fw:
            # 3  逐行遍历原始数据
            for line in fr:
                # 4  验证，去掉左右的空字符，跳过空行
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                # 5  分割 文本 和 标签索引
                text, label_idx = line.split("\t")

                # 6  把 标签id -> 标签str，加上 __label__
                label_text = id2label[int(label_idx)]
                label_name = f"__label__{label_text}"

                # 7  对 文本 做分词，根据不同的分词策略 char jieba
                if conf.use_char_segmentation:  # 如果按照 char 进行分词
                    # “我爱长沙” -> [我，爱，长，沙]
                    words_list = list(text)
                else:  # 按照 jieba 进行分词
                    # “我爱长沙” -> [我，爱，长沙]
                    words_list = jieba.lcut(text)

                # 8  拼接分词结果 “ ”.join()
                text_preprocessed = " ".join(words_list)

                # 9  拼接成完整的样本格式
                result_line = f"{label_name} {text_preprocessed}"

                # 10 写入到保存文件中
                fw.write(result_line + "\n")  # 注意要换行

                count += 1

    print(f"已完成数据处理，共 {count} 条数据，保存到 {save_path}")


# 7  分别处理 训练集 测试集 验证集
preprocess(train_data_file, preprocess_train_data_path)
preprocess(test_data_file, preprocess_test_data_path)
preprocess(dev_data_file, preprocess_dev_data_path)
