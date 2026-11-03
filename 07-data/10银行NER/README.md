---
license: other
tags:
- NER
text:
  token-classification:    
    type:
      - ner
    language:
      - zh

---

# bank命名实体识别数据集

## 数据集概述
Book9数据集是面向银行借贷领域的中文命名实体识别数据集。

### 数据集简介
本数据集包括测试集（1500），实体类型包括银行名(BANK)、客户提到的形容词(COMMENTS_ADJ)、客户提到的名词(COMMENTS_N)、银行金融产品名(PRODUCT)。

### 数据集的格式和结构
数据格式采用json格式，其中"text"对应的是原始文本，"labels"对应的是相应的表情。一个具体case的例子如下：

```
{"text": "金卡额度没上限吗", "labels": ["B-PRODUCT", "I-PRODUCT", "B-COMMENTS_N", "I-COMMENTS_N", "O", "O", "O", "O"]}

```

## 数据集版权信息

Creative Commons Attribution 4.0 International。

## 数据来源
  ```txt
    https://www.heywhale.com/mw/dataset/617969ec768f3b0017862990/file
  ```
