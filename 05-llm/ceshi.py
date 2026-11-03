import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("base_url"))

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个有用的助手，请帮助我完成我的任务。"},
        {"role": "user", "content": "你是谁？"},
    ],
    stream=False
)

result = response.choices[0].message.content
print(result)
