# coding=utf-8
"""
1、这是起始第一部分
这段代码的作用是将TensorFlow中的检查点转化为pytorch中的对应检查点
Convert BERT checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import argparse
import tensorflow as tf
import torch

from modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert


# 该方法为将检查点改为pytorch的检查点
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):

    # 加载模型参数
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    # 加载模型
    model = BertForPreTraining(config)

    # 加载检查点参数到模型中，进行处理
    # 但是有一个问题，为什么加不加返回model都能返回？？？猜测其为内部已进行处理
    load_tf_weights_in_bert(model, tf_checkpoint_path)
    print("Save PyTorch model to {}".format(pytorch_dump_path))

    # 保存pytorch的检查点
    torch.save(model.state_dict(), pytorch_dump_path)


# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default="F:/nlp/Multi_segment_extraction/chinese_L-12_H-768_A-12/bert_model.ckpt",
                        type=str,
                        help="Path the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default="F:/nlp/Multi_segment_extraction/chinese_L-12_H-768_A-12/bert_config.json",
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model.This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default="F:/nlp/Multi_segment_extraction/bert-base-chinese/pytorch_model.bin",
                        type=str,
                        help="Path to the output PyTorch model.")

    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
