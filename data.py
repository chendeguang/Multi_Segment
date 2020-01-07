# ##这个方法为加载训练以及测试语料的json文件（2019.10.17）##  这个类是没有什么新意的
# 这一个代码是用来处理数据的
#!/usr/bin/env python
# coding: utf-8

import json
from tokenization import whitespace_tokenize
from file_utils import logger
# from tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)


# 判断字符是否为控制字符
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


# 这应该是对句子进行处理
def split_by_space(paragraph_text):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens)-1)
    return doc_tokens, char_to_word_offset


# 这儿是实例化
class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s\n" % (self.qas_id)
        s += ", question_text: %s\n" % (self.question_text)
        s += ", doc_tokens: [{}]\n".format(" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: {}\n" .format(self.start_position)
        if self.end_position:
            s += ", end_position: {}\n".format(self.end_position)
        if self.is_impossible:
            s += ", is_impossible: {}\n".format(self.is_impossible)
        return s


# 这个方法为加载训练以及测试语料的json文件
def read_squad_examples(input_file, is_training, version_2_with_negative):
    """
    Read a SQuAD json file into a list of SquadExample.
    """
    with open(input_file, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:

            paragraph_text = paragraph["context"]
            doc_tokens, char_to_word_offset = split_by_space(paragraph_text)
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                # 如果进行训练，则进行下面操作
                if is_training:
                    # 为什么会执行这个if
                    if (version_2_with_negative):
                        is_impossible = qa["is_impossible"]
                    # 若多余一个问题，则出错
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError("For training, each question should have exactly 1 answer.")
                    # 如果信息不完整，我们则对信息进行相应的补充
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]

                        # 对于那些不符合规矩的文本，我们进行摒弃处理
                        # Only add answers where the text can be exactly recovered from the document. If this CAN'T
                        # happen it's likely due to weird Unicode stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible
                )
                examples.append(example)
    return examples


def read_multi_examples(input_file, is_training, version_2_with_negative=False):
    entry = [json.loads(line) for line in open(input_file, encoding='utf-8')]
    examples = []

    new_entry = entry[0]["data"]
    for paragraph in new_entry[0]["paragraphs"]:

        paragraph_text = paragraph["contexts"]
        doc_tokens, char_to_word_offset = split_by_space(paragraph_text)

        qas_id = paragraph['uid']
        question_text = paragraph['problem']
        is_impossible = False
        if is_training:
            start_position = paragraph['pos_istart']
            start_position1 = []
            for i in range(len(start_position)):
                start_position1.append(int(start_position[i]))

            end_position = paragraph['pos_iend']
            end_position1 = []
            for i in range(len(end_position)):
                end_position1.append(int(end_position[i]))

            new_start_position = [char_to_word_offset[x]for x in start_position1]
            new_end_position = [char_to_word_offset[x-1]for x in end_position1]

            orig_answer_text = paragraph['answer']
        else:
            new_start_position = None
            new_end_position = None
            orig_answer_text = None

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=new_start_position,
            end_position=new_end_position,
            is_impossible=is_impossible
        )
        examples.append(example)

    return examples

