import streamlit as st
import requests
import time
import json

st.title('新闻分类预测')

st.text_area(
    '请你输入要预测的文本：',
    '体验2D巅峰 倚天屠龙记十大创新概览'
)

if st.button('预测'):
    data = {'text': st.text_area('请你输入要预测的文本：', '体验2D巅峰 倚天屠龙记十大创新概览')}
    url = 'http://localhost:8000/predict'

    start_time = time.time()

    try:
        response = requests.post(url,json=data)
        total_time = (time.time() - start_time) * 1000  # 转换为毫秒

        if response.status_code == 200:
            result = response.json()
            st.write(f'预测结果为：{result["pred_class"]}')
            st.write(f'耗时：{total_time:.2f} ms')
        else:
            st.write(f'请求失败，状态码：{response.status_code}')
            st.write(f'响应内容：{response.text}')
    except Exception as e:
        st.write(f'请求出错：{str(e)}')



