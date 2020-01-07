"""
Official evaluation script for v1.1 of the SQuAD dataset.
note: the code can't use for me
"""
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import tensorflow as tf


def kmp(mom_string, son_string):
    # 传入一个母串和一个子串
    # 返回子串匹配上的第一个位置，若没有匹配上返回-1
    test = ''
    if type(mom_string) != type(test) or type(son_string) != type(test):
        return -1
    if len(son_string) == 0:
        return 0
    if len(mom_string) == 0:
        return -1
    # 求next数组
    next = [-1] * len(son_string)
    if len(son_string) > 1:  # 这里加if是怕列表越界
        next[1] = 0
        i, j = 1, 0
        while i < len(son_string) - 1:  # 这里一定要-1，不然会像例子中出现next[8]会越界的
            if j == -1 or son_string[i] == son_string[j]:
                i += 1
                j += 1
                next[i] = j
            else:
                j = next[j]

    # kmp框架
    m = s = 0  # 母指针和子指针初始化为0
    while(s<len(son_string) and m<len(mom_string)):
        # 匹配成功,或者遍历完母串匹配失败退出
        if s == -1 or mom_string[m] == son_string[s]:
            m += 1
            s += 1
        else:
            s = next[s]

    if s == len(son_string):  # 匹配成功
        return m - s
    # 匹配失败
    return -1


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', '', text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# 这儿计算f1的得分
def f1_score(prediction, ground_truth):
    prediction_text = ''
    for sig_prediction in prediction:
        sig_prediction = normalize_answer(sig_prediction)
        prediction_text += sig_prediction + ' '
    prediction_text = normalize_answer(prediction_text)

    ground_text = ''
    for sig_ground_truth in ground_truth:
        sig_ground_truth = normalize_answer(sig_ground_truth)
        ground_text += sig_ground_truth + ' '
    ground_text = normalize_answer(ground_text)

    prediction_text = prediction_text.strip()
    ground_text = ground_text.strip()

    prediction_tokens = prediction_text.split()
    ground_tokens = ground_text.split()

    common = Counter(prediction_tokens) & Counter(ground_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# 计算F2值
def f2_score(prediction, ground_truth):

    prediction_list = []
    for sig_prediction in prediction:
        sig_prediction = normalize_answer(sig_prediction)
        prediction_list.append(sig_prediction)

    ground_list = []
    for sig_ground_truth in ground_truth:
        sig_ground_truth = normalize_answer(sig_ground_truth)
        ground_list .append(sig_ground_truth)

    common = Counter(prediction_list) & Counter(ground_list)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f2 = (2 * precision * recall) / (precision + recall)
    return f2


# 这儿是模型的评估
def evaluate(dataset, muit_prediction):
    f1 = f2 = exact_match = number = 0

    # 这底下是一块
    orig_key = []
    orig_value = []
    for paragraph in dataset[0]["paragraphs"]:
        qas_id = paragraph['uid']
        orig_answer = paragraph['answer']
        orig_key.append(str(qas_id))

        orig_fragment = ''
        orig_answer = orig_answer.split(';')
        for value in orig_answer:
            value = value.strip()
            value = value.lower()

            if len(value) <= 2:
                continue
            else:
                # 这一点是用来处理尾巴的
                char0 = value[len(value)-1]
                if(char0 == '.' or char0 == ',' or char0 == '?' or char0 == '，' or char0 == '。'
                        or char0 == '$' or char0 == '￥'):
                    value = value[:-1]
                    char1 = value[len(value)-1]
                    if (char1 == '.' or char1 == ',' or char1 == '?' or char1 == '，' or char1 == '。'
                            or char1 == '$' or char1 == '￥'):
                        value = value[:-1]

                orig_fragment += value + ';'

        orig_value.append(orig_fragment)
    # 保存为字典类型
    orig_dict = zip(orig_key, orig_value)
    orig = dict(orig_dict)

    # 这底下是一块
    # 读取多片段中的键值对
    muit_key = []
    muit_value = []
    for k,v in muit_prediction.items():
        muit_key.append(k)

        muit_fragment = ''
        values = v.split(';')
        for value in values:
            value = value.strip()
            value = value.lower()

            if(len(value)<=2):
                continue
            else:
                # 这一点是用来处理尾巴的
                char0 = value[len(value)-1]
                if(char0 == '.' or char0 == ',' or char0 == '?' or char0 == '，' or char0 == '。'
                        or char0 == '$' or char0 == '￥'):
                    value = value[:-1]
                    char1 = value[len(value)-1]
                    if (char1 == '.' or char1 == ',' or char1 == '?' or char1 == '，' or char1 == '。'
                            or char1 == '$' or char1 == '￥'):
                        value = value[:-1]

                muit_fragment += value + ';'
        muit_value.append(muit_fragment)
    # 保存为字典类型
    muit_dict = zip(muit_key, muit_value)
    muit = dict(muit_dict)

    # 进行比较
    for orig_key, orig_value in orig.items():
        for total_key, total_value in muit.items():
            if(orig_key == total_key):
                number += 1

                # 对原始片段进行处理
                total_orig_Fragment = []
                orig_values = orig_value.split(';')
                for sig_orig_value in orig_values:
                    if len(sig_orig_value) <= 2:
                        continue
                    else:
                        sig_orig_value = normalize_answer(sig_orig_value)
                        total_orig_Fragment.append(sig_orig_value)

                # 对预测片段进行处理
                total_prediction_Fragment = []
                total_values = total_value.split(';')
                for sig_total_value in total_values:
                    if len(sig_total_value) <= 2:
                        continue
                    else:
                        sig_total_value = normalize_answer(sig_total_value)
                        total_prediction_Fragment.append(sig_total_value)

                # 计算EM值
                if(sorted(total_prediction_Fragment)== sorted(total_orig_Fragment)):
                    exact_match += 1

                # 计算F1值
                f1 += f1_score(total_prediction_Fragment, total_orig_Fragment)

                # 计算F2值
                f2 += f2_score(total_prediction_Fragment, total_orig_Fragment)

    exact_match = 100.0 * exact_match / number
    f1 = 100.0 * f1 / number
    f2 = 100.0*f2 / number

    print(exact_match)
    print(f1)
    print(f2)
    return {'exact_match': exact_match, 'f1': f1, 'f2':f2}


if __name__ == '__main__':
    flags = tf.flags
    FLAGS = flags.FLAGS
    expected_version = 'English_squad_v1.0'
    # Required parameters
    flags.DEFINE_string("dataset_file", "./SQuAD/version/prediction.json", "The config json file")
    flags.DEFINE_string("muit_prediction_file", "./SQuAD/output/predictions.json", "Multi-segment config json file")

    # 这是打开预测json文件
    with open(FLAGS.dataset_file, encoding='utf-8') as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version + ', but got dataset with v-' + dataset_json['version'], file=sys.stderr)
        dataset = dataset_json['data']

    # 这是打开预测出的json文件（多片段预测文件）
    with open(FLAGS.muit_prediction_file) as prediction_file:
        muit_prediction = json.load(prediction_file)

    # 评估函数
    evaluate(dataset, muit_prediction)





