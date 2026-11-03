# 1  导包
import streamlit as st  # 绘制页面
import requests  # 发送请求
import time

# todo 在命令行执行
#  streamlit run streamlit_app.py

# 2  页面标题
st.title("新闻分类预测")

# 3  输入框
text_input = st.text_area(
    "请你输入要预测的文本:",  # 提示词
    "体验2D巅峰 倚天屠龙记十大创新概览"  # 默认句子
)

# 4  预测按钮
if st.button("预测"):
    # 5  构造 数据data 请求url
    data = {"text": text_input}
    url = "http://127.0.0.1:8000/predict"

    # 6  记录时间
    start_time = time.time()

    # 7  发送请求
    try:
        response = requests.post(url, json=data)
        total_time = (time.time() - start_time) * 1000
        # 8  解析响应 判断 状态码 200
        if response.status_code == 200:
            result = response.json()
            label = result["pred_class"]
            st.success(f"预测分类: {label}")  # 成功格式的信息
            st.info(f"请求耗时: {total_time:.2f}ms")  # 一般信息
        else:
            st.error(f"请求失败: {response.status_code}")  # 失败信息格式

    except Exception as e:
        st.error(f"请求出错: {str(e)}")

# 9  运行提示
st.write("请你确保各字段一致，端口号在 8000 运行")
