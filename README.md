# transformer_model
基于原生transformer训练各类模型

一. en_to_zh: 英文-中文翻译模型
    1. 训练:进入en_to_zh目录 nohup python -u en_to_zh_train.py >> train.log 2>&1 &
    2. 用训练好的模型推理翻译：python en_to_zh_infer_demo.py
