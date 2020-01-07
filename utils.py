# 顾名思义，这是一个工具类，但是这个工具类是用来干嘛的呢
#!/usr/bin/env python
# coding: utf-8

import math
import json
import logging
import collections

from tokenization import BasicTokenizer

logger = logging.getLogger(__name__)


def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
        if c == " ":
            continue
        ns_to_s_map[len(ns_chars)] = i
        ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """
    Project the tokenized prediction back to the original text.
    """
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score/total_sum)
    return probs


# 对单标签的什么进行预测
# Write final predictions to the json file and log-odds of null if needed.
def write_predictions_single_labeling(all_examples, all_features, all_results,
                                      n_best_size, max_answer_length, do_lower_case,
                                      output_prediction_file, output_nbest_file,
                                      output_null_log_odds_file, verbose_logging,
                                      version_2_with_negative, null_score_diff_threshold):

    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple("PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            # TODO add decode
            thresh = 0
            max_start_score = -1000
            end_indexes = []
            start_indexes = []
            max_start_idx = 0
            for idx, (p, start, end) in enumerate(zip(feature.segment_ids[:-1], result.start_logits, result.end_logits)):
                if p == 0:
                    continue
                if p == 1 and feature.segment_ids[idx + 1] == 0:
                    break
                if start > thresh:
                    start_indexes.append(idx)
                if result.start_logits[idx] > max_start_score:
                    max_start_idx = idx

            if len(start_indexes) == 0:
                start_indexes.append(max_start_idx)

            # add a fake index for boundary case
            start_indexes.append(10000)
            assert len(start_indexes) >= 2, print(start_indexes)
            start_index = start_indexes[0]
            accumulate_logits = 0

            for i, index in enumerate(start_indexes[:-1]):
                accumulate_logits += result.start_logits[index]
                if index + 1 == start_indexes[i+1]:
                    continue
                end_index = index
                prelim_predictions.append(_PrelimPrediction(feature_index=feature_index, start_index=start_index, end_index=end_index, start_logit=accumulate_logits, end_logit=0))
                start_index = start_indexes[i + 1]
                accumulate_logits = 0
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        seen_predictions = {}
        nbest = []
        _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)

                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # In very rare edge cases could have no valid predictions.we create a nonce prediction in this case to avoid.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = best_non_null_entry.text
        # all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # with open(output_nbest_file, "w") as writer:
    #     writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


# 这儿与上一个有啥区分
# Write final predictions to the json file and log-odds of null if needed.
def write_predictions_couple_labeling(all_examples, all_features, all_results, n_best_size,
                                      max_answer_length, do_lower_case, output_prediction_file, output_nbest_file,
                                      output_null_log_odds_file, verbose_logging, version_2_with_negative, null_score_diff_threshold):

    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    _PrelimPrediction = collections.namedtuple("PrelimPrediction",["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            # TODO add decode
            start_indexes = []
            end_indexes = []
            thresh = 0
            for idx, (p, start, end) in enumerate(zip(feature.segment_ids[:-1], result.start_logits, result.end_logits)):
                if p == 0:
                    continue
                if p == 1 and feature.segment_ids[idx + 1] ==  0:
                    break
                if start > thresh:
                    start_indexes.append(idx)
                if end > thresh:
                    end_indexes.append(idx)

            # for start_index in start_indexes:
            #     for end_index in end_indexes:
            #         if end_index >= start_index:
            #             prelim_predictions.append(
            #                 _PrelimPrediction(feature_index=feature_index,
            #                                   start_index=start_index,
            #                                   end_index=end_index,
            #                                   start_logit=result.start_logits[start_index],
            #                                   end_logit=result.end_logits[end_index]))
            #             break

            if len(start_indexes) == len(end_indexes):
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if end_index >= start_index:
                            prelim_predictions.append(
                                _PrelimPrediction(feature_index=feature_index,
                                                  start_index=start_index,
                                                  end_index=end_index,
                                                  start_logit=result.start_logits[start_index],
                                                  end_logit=result.end_logits[end_index]))
                            break
            else:
                start_indexes_prob = []
                end_indexes_prob = []
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if end_index >= start_index:
                            start_indexes_prob.append(start_index)
                            end_indexes_prob.append(end_index)
                            break

                for i in range(len(end_indexes_prob)):
                    if i != len(end_indexes_prob) - 1:
                        if end_indexes_prob[i] != end_indexes_prob[i + 1]:
                            prelim_predictions.append(
                                _PrelimPrediction(feature_index=feature_index,
                                                  start_index=start_indexes_prob[i],
                                                  end_index=end_indexes_prob[i],
                                                  start_logit=result.start_logits[start_indexes_prob[i]],
                                                  end_logit=result.end_logits[end_indexes_prob[i]]))
                            continue
                    else:
                        prelim_predictions.append(
                            _PrelimPrediction(feature_index=feature_index,
                                              start_index=start_indexes_prob[i],
                                              end_index=end_indexes_prob[i],
                                              start_logit=result.start_logits[start_indexes_prob[i]],
                                              end_logit=result.end_logits[end_indexes_prob[i]]))

        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index >= 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # In very rare edge cases we could have no valid predictions.we ust create a nonce prediction to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        best_non_null_entry = []
        for entry in nbest:
            if entry.text:
                best_non_null_entry.append(entry)

        predictions_text = ""
        for i in range(len(best_non_null_entry)):
            if (i==len(best_non_null_entry)-1):
                predictions_text = predictions_text + best_non_null_entry[i].text
            else:
                predictions_text = predictions_text + best_non_null_entry[i].text + ";"

        all_predictions[example.qas_id] = predictions_text

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 这个方法为下一个方法所调用
def _get_best_indexes(logits, n_best_size):
    """
    Get the n-best logits from a list.
    """
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


# Write final predictions to the json file and log-odds of null if needed.
def write_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case,
                      output_prediction_file, output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):

    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple("PrelimPrediction",["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(_PrelimPrediction(feature_index=feature_index, start_index=start_index, end_index=end_index, start_logit=result.start_logits[start_index], end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(_PrelimPrediction(feature_index=min_null_feature_index, start_index=0, end_index=0, start_logit=null_start_logit, end_logit=null_end_logit))

        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
        _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if entry.text:
                best_non_null_entry.append(entry)

        probs = _compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            predictions_text = ""
            for i in range(len(best_non_null_entry)):
                if (i == len(best_non_null_entry) - 1):
                    predictions_text = predictions_text + best_non_null_entry[i].text
                else:
                    predictions_text = predictions_text + best_non_null_entry[i].text + ";"

            all_predictions[example.qas_id] = predictions_text
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # with open(output_nbest_file, "w") as writer:
    #     writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    #
    # if version_2_with_negative:
    #     with open(output_null_log_odds_file, "w") as writer:
    #         writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


# # 该方法在此处没有任何作用
# def recover_original_text(feature, example, pred):
#     if pred.start_index > 0:
#         tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
#         orig_doc_start = feature.token_to_orig_map[pred.start_index]
#         orig_doc_end = feature.token_to_orig_map[pred.end_index]
#         orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
#         tok_text = " ".join(tok_tokens)
#
#         # De-tokenize WordPieces that have been split off.
#         tok_text = tok_text.replace(" ##", "")
#         tok_text = tok_text.replace("##", "")
#
#         # Clean whitespace
#         tok_text = tok_text.strip()
#         tok_text = " ".join(tok_text.split())
#         orig_text = " ".join(orig_tokens)
#
#         final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
#     else:
#         final_text = ""
#
#     return _NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit)
#
