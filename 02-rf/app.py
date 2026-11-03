# 1  导包
from flask import Flask, request, jsonify
from predictor import predict  # 完成预测

# 2  创建一个 app
app = Flask(__name__)


# 3  创建 路径函数，来完成预测功能
# 装饰器：请求路径 + 请求方式
# todo 请求方式 get 读取；post 创建；put 更新；delete 删除
# @app.route("/predict", methods=["post"])
@app.post("/predict")
def predict_api():  # 路径函数
    # 1  获取请求输入
    data = request.get_json()  # data {text: xxx}

    # 2  验证环节: 拿到的输入是否合法
    if not data or "text" not in data:
        return jsonify({"error": "Missing text field in json"}), 400

    # 3  调用预测函数，完成预测
    result = predict(data)  # result dict

    # 4  返回 json
    return jsonify(result)


if __name__ == '__main__':
    # 4  启动 app
    app.run(
        host="0.0.0.0",  # ip
        port=8000,  # 端口号
        debug=True,  # 调试模式，显示详细报错信息
    )
