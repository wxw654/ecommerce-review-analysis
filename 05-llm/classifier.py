# 1  导包
import os
import time

# load_dotenv 加载 .env 环境变量
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from tqdm import tqdm

from config import Config

# 拿到配置信息
conf = Config()
# 加载环境变量 .env，如果在同一级目录，不需要写路径
load_dotenv(override=True)

# 2  标签和 id 之间的转换：id2name {id:类别}； name2id {类别：id} -》 todo 通过加载类别文件来实现
with open(conf.class_path, "r", encoding="utf-8") as f:
    id2name = {idx: line.strip() for idx, line in enumerate(f)}

# 为了得到 name2id：遍历 id2name 的 kv 对，调换顺序
name2id = {name: idx for idx, name in id2name.items()}
print(id2name)
print(name2id)


# 3  单条样本预测
# text 具体的一条样本
def model2pred(text):
    # 1  构建代理 client
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("base_url")
    )

    # 2  定义提示词
    system_prompt = '''
    你是一个优秀的文本分类师，能把给定的用户query划分到正确的类目中。现在请你根据给定信息和要求，为给定用户query，从备选类目中选择最合适的类目。

    下面是“参考案例”即被标注的正确结果，可供参考：
    文本：中国国家乒乓球队击败日本
    类别：sports

    备选类目：
    finance,realty,stocks,education,science,society,politics,sports,game,entertainment

    请注意：
    1. 用户query所选类目，仅能在【备选类目】中进行选择，用户query仅属于一个类目。
    2. “参考案例”中的内容可供推理分析，可以仿照案例来分析用户query的所选类目。
    3. 请仔细比对【备选类目】的概念和用户query的差异。
    4. 如果用户query也不属于【备选类目】中给定的类目，或者比较模糊，请选择“拒识”。
    5. 请在“文本类别:”后回复结果，不需要说明理由。
    '''

    # 3  记录当前时间
    start_time = time.time()

    # 4  具体调用预测
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        stream=False
    )

    # 5  解析结果，统计耗时
    result = response.choices[0].message.content
    # 耗时 ms
    use_time = (time.time() - start_time) * 1000

    # 6  返回结果
    return result, use_time


# 4  整个验证集的预测评估
# model2dev 需要调用 model2pred 来实现整个 dev 数据集 的预测评估
def model2dev(dev_data):
    # 1  创建两个 list -> 用来保存 预测结果 和 真实标签
    preds, labels = [], []

    # 2  记录时间
    start_time = time.time()

    # 3  读取验证集
    with open(dev_data, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="数据预测中...")):
            # 4  提取文本和真实标签
            text, label = line.strip().split("\t")

            # 5  执行模型预测 model2pred
            result, _ = model2pred(text)

            # 6  把预测类别转为类别编号
            name = result.strip().split(":")[1]
            pred = name2id[name.strip()]

            # 7  分别保存预测值和标签值
            preds.append(int(pred))
            labels.append(int(label))

            if idx == 10:
                break

    # 8  计算评估指标
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="micro")
    precision = precision_score(labels, preds, average="micro")
    recall = recall_score(labels, preds, average="micro")
    use_time = (time.time() - start_time) * 1000

    # 9  返回评估指标
    return accuracy, f1, precision, recall, use_time


if __name__ == '__main__':
    # text = "本科未录取还有这些路可以走"
    # result, use_time = model2pred(text)
    # print(f"预测结果为:\t{result}")
    # print(f"请求耗时:\t{use_time}ms")

    model2dev(conf.dev_path)
