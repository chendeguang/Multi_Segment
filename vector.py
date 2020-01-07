#!/usr/bin/env python
# coding: utf-8

import logging
import collections
from tqdm import tqdm

logger = logging.getLogger(__name__)


# 将单词片段进行切分
def wordpiece_split(tokens, tokenizer):
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    return tok_to_orig_index, orig_to_tok_index, all_doc_tokens


# 构造答案对
def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """
    Returns tokenized answer spans that better match the annotated answer.
    """
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


# 检查最大的什么东西呢
def _check_is_max_context(doc_spans, cur_span_index, position):
    """
    Check if this is the 'max context' doc span for the token.
    """
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


# 构造对象
class InputFeatures(object):
    """
    A single set of features of data.
    """
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_vector=None,
                 end_vector=None,
                 content_vector=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_vector = start_vector
        self.end_vector = end_vector
        self.content_vector = content_vector
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    # def __str__(self):
    #     buff = []
    #     buff.append("*** Example ***")
    #     buff.append("unique_id: %s" % (self.unique_id))
    #     buff.append("example_index: %s" % (self.example_index))
    #     buff.append("doc_span_index: %s" % (self.doc_span_index))
    #     buff.append("tokens: %s" % " ".join(self.tokens))
    #     buff.append("token_to_orig_map: %s" % " ".join(["%d:%d" % (x, y) for (x, y) in self.token_to_orig_map.items()]))
    #     buff.append("token_is_max_context: %s" % " ".join(["%d:%s" % (x, y) for (x, y) in self.token_is_max_context.items()]))
    #     buff.append("input_ids: %s" % " ".join([str(x) for x in self.input_ids]))
    #     buff.append("input_mask: %s" % " ".join([str(x) for x in self.input_mask]))
    #     buff.append("segment_ids: %s" % " ".join([str(x) for x in self.segment_ids]))
    #     if isinstance(self.start_position, list):
    #         answer_text = [" ".join(self.tokens[s:e+1]) for s,e in zip(self.start_position, self.end_position)]
    #     else:
    #         answer_text = " ".join(self.tokens[self.start_position:(self.end_position + 1)])
    #     buff.append("start_position: {}".format(self.start_position))
    #     buff.append("end_position: {}".format(self.end_position))
    #     buff.append('start_vector: {}'.format(self.start_vector))
    #     buff.append('end_vector: {}'.format(self.end_vector))
    #     buff.append('content_vector: {}'.format(self.content_vector))
    #     buff.append("answer: %s" % (answer_text))
    #     return '\n'.join(buff)
    #
    # def __repr__(self):
    #     return self.__str__()


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    """
    Loads a data file into a list of `InputBatch`s.
    """
    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(tqdm(examples)):
        query_tokens = tokenizer.tokenize(example.question_text)

        # 如果token长度大于了最大长度，则对其进行截取处理
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # 这个处理是存在问题的
        tok_to_orig_index, orig_to_tok_index, all_doc_tokens = wordpiece_split(example.doc_tokens, tokenizer)

        tok_start_position = None
        tok_end_position = None
        # 判断是否进行训练，进而进行处理
        if is_training==True and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        # 这又是一个什么判断语句
        if is_training==True and not example.is_impossible and isinstance(example.start_position, list):
            tok_start_position = [orig_to_tok_index[x] for x in example.start_position]
            tok_end_position = []
            for x in example.end_position:
                if x < len(example.doc_tokens) - 1:
                    tok_end_position.append(orig_to_tok_index[x + 1] - 1)
                else:
                    tok_end_position.append(len(all_doc_tokens) - 1)
        # 以上三个if都具有
        if is_training==True and not example.is_impossible and not isinstance(example.start_position, list):
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        # We can have documents that are longer than the maximum sequence length.To deal with this we do a sliding
        # window approach, where we take chunks of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            # only one span
            break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):

            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_vector = [0] * max_seq_length
            end_vector = [0] * max_seq_length
            content_vector = [0] * max_seq_length

            start_position = None
            end_position = None
            # start_position = []
            # end_position = []
            if is_training and isinstance(tok_start_position, list):
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                doc_offset = len(query_tokens) + 2
                for s in tok_start_position:
                    if s < doc_start or s > doc_end:
                        continue
                    start_vector[doc_offset + s - doc_start] = 1
                for e in  tok_end_position:
                    if e > doc_end:
                        continue
                    end_vector[doc_offset + e - doc_span.start] = 1
                for s, e in zip(tok_start_position, tok_end_position):
                    if not (s >= doc_start and e <= doc_end):
                        continue
                    for i in range(s, e+1):
                        content_vector[doc_offset + i - doc_span.start] = 1
                # for s, e in zip(tok_start_position, tok_end_position):
                #     start_position.append(s - doc_start + doc_offset)
                #     end_position.append(e - doc_start + doc_offset)
                for s, e in zip(tok_start_position, tok_end_position):
                    start_position = s - doc_start + doc_offset
                    end_position = e - doc_start + doc_offset
                    break  # 这儿的break我实在弄不明白是用来干啥的

            if is_training and not example.is_impossible and not isinstance(tok_start_position, list):
                # For training, if our document chunk does not contain an annotation we throw it out.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_vector=start_vector,
                    end_vector=end_vector,
                    content_vector=content_vector,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible)
            )
            unique_id += 1

    return features

