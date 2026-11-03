# 1  导包
import requests  # 发送请求
import time

# 2  定义接口地址 url = ip + port + 请求地址
url = "http://127.0.0.1:8000/predict"

# 3  构造请求数据
data = {"text": "同步A股首秀：港股缩量回调"}

# 4  记录开始时间
start_time = time.time()

# 5  发送 post 请求
try:
    response = requests.post(url, json=data)
    # 6  计算耗时
    total_time = (time.time() - start_time) * 1000  # 单位 毫秒
    # 7  检查响应状态 200
    if response.status_code == 200:
        result = response.json()
        print(f"预测结果为: {result['pred_class']}")
    else:
        print(f"请求失败: {response.status_code}")

except Exception as e:
    print(f"请求出错: {str(e)}")
