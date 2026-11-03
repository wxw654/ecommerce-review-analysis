---
license: Apache License 2.0
domain: NLP
task:
  - text-classification
---

## 数据集简介
该数据集包含基于道德基础分类的新闻文章，适用于中文的多标签文本分类任务。每篇新闻文章根据其道德倾向进行标注，分为实用（Pragmatism）/ 理想（Idealism）”、“责任（Responsibility）/ 利益（Profit）”、“创新（Innovation）/ 守旧（Conservatism）三个标签。

## 许可证
该数据集的使用许可为 Apache License 2.0，您可以自由使用、修改和分发该数据集，但需遵循许可协议的规定。

## 数据集格式
数据集以 CSV 格式提供，包含三个主要字段：
- **system**：表示对模型的初始命令。
- **query**：表示新闻文章的文本内容。
- **response**：表示模型应返回的新闻文章的分类标签，支持多标签分类。

## 任务和语言
- **任务类型**：文本分类（多标签）
- **语言**：中文

## 下载方法
您可以通过以下两种方式下载数据集：
### 使用 ModelScope SDK 下载
```python
from modelscope.msdatasets import MsDataset
dataset = MsDataset.load('wowhaha/moral-foundation-news_2000')

