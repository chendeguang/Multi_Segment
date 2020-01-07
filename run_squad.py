# coding=utf-8
"""
Run BERT on SQuAD.
"""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import logging
import os
import random
import sys
from io import open
from pprint import pprint  # 这样具有美观打印效果；参考网址：https://blog.csdn.net/u010105243/article/details/53224551
import math
import json
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from modeling import BertForQuestionAnswering
from optimization import BertAdam, WarmupLinearSchedule
from tokenization import BertTokenizer

from vector import convert_examples_to_features
from data import read_squad_examples, read_multi_examples   # 加载read_squad_examples
from utils import write_predictions, write_predictions_couple_labeling, write_predictions_single_labeling

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)
RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


def main():
    parser = argparse.ArgumentParser()
    # # 必要参数
    parser.add_argument('--task', default='multi', type=str, help='Task affecting load data and vectorize feature')
    parser.add_argument('--loss_type', default='double', type=str, help='Select loss double or single, only for multi task')  # 针对multi任务才有效
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased,bert-large-uncased, "
                             "bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,bert-base-chinese,"
                             "bert-base-multilingual-cased.")   # 选择预训练模型参数
    parser.add_argument("--debug", default=False, help="Whether run on small dataset")  # 正常情况下都应该选择false
    parser.add_argument("--output_dir", default="./SQuAD/output/", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # # 其他参数
    parser.add_argument("--train_file",
                        default="./SQuAD/version/train.json",
                        type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file",
                        default="./SQuAD/version/prediction.json",
                        type=str, help="SQuAD json for predictio ns. E.g., dev-v1.1.json or test-v1.1.json")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will be "
                             "truncated to this length.")

    # # 控制参数
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_predict", default=True, help="Whether to run eval on the dev set.")

    parser.add_argument("--train_batch_size", default=18, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=18, type=int, help="Total batch size for predictions.")

    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated.This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", default=False,
                        help="If true, all of the warnings related to data processing will be printed.A number of "
                             "warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda", default=False, help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', default=False, help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative', default=False,
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    args = parser.parse_args()

    # if是采用单机形式，else采用的是分布式形式；因为我们没有分布式系统，所以采用单机多GPU的方式进行训练10.24
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='hierarchical_copy')

    # 以下三句话的意义不是很大，基本操作这一部分是日志的输出形式10.24
    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device:{}, n_gpu:{}, distributed training:{}, 16-bits training:{}".format(device, n_gpu, bool(args.local_rank != -1), args.fp16))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    # 以下几行均是用来设置参数10.24
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    random.seed(args.seed)  # 设置随机种子
    np.random.seed(args.seed)  # 设置随机种子
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    if n_gpu > 0:   # 如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子
        torch.cuda.manual_seed_all(args.seed)

    # 以下三句又是基本操作，意义不大10.24
    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")
    if args.do_train:
        if not args.train_file:
            raise ValueError("If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")

    # 以下2句是用来判断output_dir是否存在，若不存在，则创建即可（感觉有这个东西反而不太好，因为需要空文件夹）10.24
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 这个东西是用来干啥的（从tokenization中读取，对Tokenizer进行初始化操作）10.24
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # 从data中读取数据的方式，一种是单队列的读取方式，另一种是多通道读取方式10.24
    if args.task == 'squad':
        read_examples = read_squad_examples
    elif args.task == 'multi':
        read_examples = read_multi_examples

    # 用来加载训练样例以及优化的步骤10.24
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_examples(input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)
        if args.debug:
            train_examples = train_examples[:100]
        num_train_optimization_steps = \
            int(len(train_examples)/args.train_batch_size/args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # 模型准备中ing10.24
    model = BertForQuestionAnswering.from_pretrained(
        args.bert_model,
        cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    )

    # model = torch.nn.DataParallel(model).cuda()
    # 判断是否使用float16编码10.24
    if args.fp16:
        # model.half().cuda()
        model.half()
        # 将模型加载到相应的CPU或者GPU中10.24
    model.to(device)

    # 配置优化器等函数10.24
    if args.do_train:
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        if args.fp16:
            try:
                # from apex.optimizers import FP16_Optimizer
                from apex.fp16_utils import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, bias_correction=True)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion, t_total=num_train_optimization_steps)
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    # 进行模型的拟合训练10.24
    global_step = 0
    if args.do_train:
        # 训练语料的特征提取
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)

        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        all_start_vector = torch.tensor([f.start_vector for f in train_features], dtype=torch.float)
        all_end_vector = torch.tensor([f.end_vector for f in train_features], dtype=torch.float)
        all_content_vector = torch.tensor([f.content_vector for f in train_features], dtype=torch.float)

        # # 替换的内容all_start_positions以及all_end_positions
        # all1_start_positions = []
        # for i in range(len(train_features)):
        #     for j in range(len(train_features[i].start_position)):
        #         all1_start_positions.append(train_features[i].start_position[j])
        # all_start_positions = torch.tensor([k for k in all1_start_positions], dtype=torch.long)
        # all1_end_positions = []
        # for i in range(len(train_features)):
        #     for j in range(len(train_features[i].end_position)):
        #         all1_end_positions.append(train_features[i].end_position[j])
        # all_end_positions = torch.tensor([k for k in all1_end_positions], dtype=torch.long)
        # ####################################################################

        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_start_positions,
            all_end_positions, all_start_vector, all_end_vector, all_content_vector
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)  # 随机采样器
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            # 每次都叫他进行分发，这样的话，就可以进行多GPU训练
            model = torch.nn.DataParallel(model).cuda()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):

                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions, start_vector, end_vector, content_vector = batch

                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions, start_vector, end_vector, content_vector, args.loss_type)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    print("loss率为：{}".format(loss))
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            print("\n")
            print(ep)
            output_model_file = os.path.join(args.output_dir, str(ep) + WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, str(ep) + CONFIG_NAME)

            torch.save(model.state_dict(), output_model_file)
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            model.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)

    # 这个是用来加载进行微调调好后的代码以方便进行预测10.25
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForQuestionAnswering.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForQuestionAnswering.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    # 再次将GPU加入10.25
    model.to(device)

    # 这部分就是进行相应的预测（用于生成预测文件）
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = \
            read_examples(input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
        if args.debug:
            eval_examples = eval_examples[:100]
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False
        )

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

        middle_result = os.path.join(args.output_dir, 'middle_result.pkl')
        pickle.dump([eval_examples, eval_features, all_results], open( middle_result, 'wb'))

        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")

        if (args.loss_type == 'double'):
            write_predictions_couple_labeling(
                eval_examples, eval_features, all_results, args.n_best_size, args.max_answer_length,
                args.do_lower_case, output_prediction_file, output_nbest_file, output_null_log_odds_file,
                args.verbose_logging, args.version_2_with_negative, args.null_score_diff_threshold
            )
        elif (args.loss_type == 'single'):
            write_predictions_single_labeling(
                eval_examples, eval_features, all_results, args.n_best_size,
                args.max_answer_length, args.do_lower_case, output_prediction_file, output_nbest_file,
                output_null_log_odds_file, args.verbose_logging, args.version_2_with_negative, args.null_score_diff_threshold
            )
        elif (args.loss_type == 'origin') or (args.task == 'multi' and args.loss_type == 'squad'):
            write_predictions(
                eval_examples, eval_features, all_results, args.n_best_size,
                args.max_answer_length, args.do_lower_case, output_prediction_file, output_nbest_file,
                output_null_log_odds_file, args.verbose_logging, args.version_2_with_negative, args.null_score_diff_threshold
            )
        else:
            raise ValueError('{} dataset and {} loss is not support'.format(args.task, args.loss_type))


if __name__ == "__main__":
    main()
