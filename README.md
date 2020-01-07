# Multi_Segment 基于BERT的多片段阅读理解模型

## 软件版本
* pytorch 0.4.1 

## 预训练模型
* 本文基于两种预训练模型，bert-base-uncased与bert-large-uncased两种预训练模式

## 训练
* 直接run_squad.py，其中参数的设置
  --task：multi与squad（multi是读取多片段json语料的文件、squad是读取单片段json语料的文件）
  --loss_type: loss类型，在进行loss时域答案的预测过程中均会用到
  --bert_mode: 此处需要设置
  --output_dir: 输出地址
  --train_file: 训练语料
  --predict_file: 预测语料

## 预测
  *直接运行evaluate_double.py即可
  
  
# 说明
  本代码原为在苏立新代码基础上(https://github.com/lixinsu/multi_span),删除了其中冗余的部分，并在训练与预测阶段做了一定优化。
