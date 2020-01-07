# BertTokenizer（2019.10.17）
# coding=utf-8
"""
Tokenization classes.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open

from file_utils import cached_path
logger = logging.getLogger(__name__)


# 参数
PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "./bert-base-uncased/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "F:/MuitSpan/Multi_Segment_Multi_Classifier/bert-large-uncased/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "F:/nlp/Multi_segment_extraction/bert-base-chinese/bert-base-chinese-vocab.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
}
VOCAB_NAME = 'bert-base-uncased-vocab.txt'  # bert-large-uncased-vocab.txt  bert-base-uncased-vocab.txt


# 该方法为祛除两边以及按空格进行切分10.24
def whitespace_tokenize(text):
    """
    Runs basic whitespace cleaning and splitting on a piece of text.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


# 加载词向量10.24
def load_vocab(vocab_file):
    """
    Loads a vocabulary file into a dictionary.
    """
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


# 这个类的操作是对Token进行基础的处理
class BasicTokenizer(object):
    def __init__(self, do_lower_case=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """
        Constructs a BasicTokenizer.
        Args: do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    # 除本方法以外，其余所有方法都是为该方法服务的
    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            # if self.do_lower_case and token not in self.never_split:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    # 这是_tokenize_chinese_chars方法中的调用，用于判断字符是否和服规范
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
                (cp >= 0x3400 and cp <= 0x4DBF) or
                (cp >= 0x20000 and cp <= 0x2A6DF) or
                (cp >= 0x2A700 and cp <= 0x2B73F) or
                (cp >= 0x2B740 and cp <= 0x2B81F) or
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or
                (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True

        return False


# 这个类是将一个单词切分为几个能够识别的简单单词（但是对于汉语来说，我个人觉得是多余的）
class WordpieceTokenizer(object):
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


# 这才是BERT的关键"""Runs end-to-end tokenization: punctuation splitting + wordpiece"""
class BertTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
          self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    # 该方法是BertTokenizer的核心方法，其调用了BasicTokenizer与WordpieceTokenizer的核心方法tokenize
    def tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 这个方法的作用是将tokens转化为ids（这个方法与下一个方法是互逆）
    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length for BERT model({}>{})."
                "Running this sequence through BERT will result in indexing errors".format(len(ids), self.max_len))
        return ids

    # 这个方法的作用是将ids转化为tokens
    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    # 保存vocab.txt
    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   "Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return vocab_file

    # 该方法是用来对vocab.txt进行处理
    # #classmethod修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的 cls 参数，
    # #可以来调用类的属性，类的方法，实例化对象等
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        # 从哪儿加载文件10.24
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
            if '-cased' in pretrained_model_name_or_path and kwargs.get('do_lower_case', True):
                logger.warning(
                    "The pre-trained model you are loading is a cased model but you have not set `do_lower_case` to "
                    "False.We are setting `do_lower_case=False` for you but you may want to check this behavior.")
                kwargs['do_lower_case'] = False
            elif '-cased' not in pretrained_model_name_or_path and not kwargs.get('do_lower_case', True):
                logger.warning(
                    "The pre-trained model you are loading is an uncased model but you have set `do_lower_case` to "
                    "False.We are setting `do_lower_case=True` for you but you may want to check this behavior.")
                kwargs['do_lower_case'] = True
        else:
            vocab_file = pretrained_model_name_or_path

        # 判断是否有这个文件，进而进行相应处理10.24
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)

        # redirect to the cache, if necessary
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}).We assumed '{}' was a path or url but couldn't "
                "find any file associated to this path or"
                " url.".format(pretrained_model_name_or_path, ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()), vocab_file)
            )
            return None

        # 这儿是打印从哪儿进行加载
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(vocab_file, resolved_vocab_file))

        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)

        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)

        return tokenizer


# --------------------------------这些都是些判定方法------------------------------------------- #
def _is_control(char):
    """
    Checks whether `chars` is a control character.
    """
    # These are technically control characters but we count them as whitespace characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


# 该方法是用于判断控制字符干啥用的
def _is_whitespace(char):
    """
    Checks whether `chars` is a whitespace character.
    """
    # \t, \n, and \r are technically contorl characters but we treat them as whitespace since they are generally.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_punctuation(char):
    """
    Checks whether `chars` is a punctuation character.
    """
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation. Characters such as "^", "$", and "`"  are not
    # in the Unicode Punctuation class but we treat them as punctuation anyways, for consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
