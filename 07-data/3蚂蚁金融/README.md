---
pretty_name: AFQMC
license: CC-BY-4.0
source_datasets:
- original 
tags:
- Ant
size_scale:
- 100K<n<1M

text:
  text-classification:
    language:
      - zh
    size_scale:
      - 10k-1m

---


# 概述

AFQMC（Ant Financial Question Matching Corpus）蚂蚁金融语义相似度数据集，用于问题相似度计算。即：给定客服里用户描述的两句话，用算法来判断是否表示了相同的语义。


# 数据集描述

本数据集包括训练集（34334）验证集（4316）测试集（3861）。其中，每一条数据有三个属性，分别是句子1，句子2，句子相似度标签。

其中label标签中，"1" ：表示sentence1和sentence2的含义类似；"0"：表示sentence1和sentence2的含义不同。

例子： {"sentence1": "双十一花呗提额在哪", "sentence2": "里可以提花呗额度", "label": "0"} 

数据集可从https://www.cluebenchmarks.com/introduce.html 获取


# 范例
A. 语义不同：
{"sentence1":"双十一花呗提额在哪","sentence2":"哪里可以提花呗额度","label":"0"}

B. 语义相同：
{"sentence1":"花呗如何还款","sentence2":"花呗怎么还款","label":"1"}

