from __future__ import absolute_import, division, print_function
from builtins import exit, print, type
import argparse
import csv
from operator import index
import os
import random
import pickle
import sys
import numpy as np
from typing import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from torch.utils import data

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from bert import SPAMN_BertForSequenceClassification
from xlnet import SPAMN_XLNetForSequenceClassification

#from roberta import SPAMN_RobertaForSequenceClassification
from roberta1 import SPAMN_RobertaForSequenceClassification
from argparse_utils import str2bool, seed
from global_configs import ACOUSTIC_DIM, ALPHA, VISUAL_DIM, DEVICE, BETA, GAMA, BL

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=60)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base", 'roberta-large'],
    default="bert-base-uncased",
)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default="random")

parser.add_argument("--best_acc", type=float, default=0.1)

args = parser.parse_args()

data_no = {}
word_no = {}
data_index = 0



def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id, data_no, sentence):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.data_no = data_no,
        self.sentence = sentence


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob, backbone_layer):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob
        self.backbone_layer = backbone_layer


def save_model(model, name=''):
    torch.save(model, f'pretrains/{name}.pt')

def load_model(name=''):
    model = torch.load(f'pretrains/{name}.pt')
    return model


def attn_map(name, data):
    random_array = data
    figure = plt.figure()
    axes = figure.add_subplot(111)
    # using the matshow() function
    caxes = axes.matshow(random_array, interpolation='nearest')
    figure.colorbar(caxes)
    axes.set_xticklabels([''])
    axes.set_yticklabels([''])
    #plt.show()
    plt.savefig(f'./img/{name}.jpg')

def convert_to_features(examples, max_seq_length, tokenizer):
    features = []
    file_obj = open(f'result/{args.model}_tokens.txt', 'w')
    print(type(examples))
    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic), label_id, segment = example
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))
        # Check inversion
        assert len(tokens) == len(inversions)
        print(len(tokens), visual.shape, acoustic.shape)
        print(type(inversions), inversions)
        
        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)
        print(visual.shape, acoustic.shape)
    
        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        if args.model == "bert-base-uncased":
            prepare_input = prepare_bert_input
        elif args.model == "xlnet-base-cased":
            prepare_input = prepare_xlnet_input
        elif args.model == "roberta-base":
            prepare_input = prepare_roberta_input
        elif args.model == "roberta-large":
            prepare_input = prepare_roberta_input
        
        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer,file_obj=file_obj
        )
        

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
                data_no=segment,
                sentence=words
            )
        )
    file_obj.close()
    return features


def prepare_bert_input(tokens, visual, acoustic, tokenizer, file_obj=None):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]
    

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    file_obj.writelines(str(tokens) + '\n')
    file_obj.writelines(str(input_ids) + '\n')
    return input_ids, visual, acoustic, input_mask, segment_ids

def prepare_roberta_input(tokens, visual, acoustic, tokenizer, file_obj=None):
    BOS = tokenizer.bos_token
    EOS = tokenizer.eos_token
    tokens = [BOS] + tokens + [EOS]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    file_obj.writelines(str(tokens) + '\n')
    file_obj.writelines(str(input_ids) + '\n')
    return input_ids, visual, acoustic, input_mask, segment_ids

def prepare_xlnet_input(tokens, visual, acoustic, tokenizer, file_obj=None):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    PAD_ID = tokenizer.pad_token_id

    # PAD special tokens
    tokens = tokens + [SEP] + [CLS]
    #print(tokens)
    audio_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, audio_zero, audio_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual, visual_zero, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(tokens) - 1) + [2]

    pad_length = (args.max_seq_length - len(segment_ids))

    # then zero pad the visual and acoustic
    audio_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((audio_padding, acoustic))

    video_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((video_padding, visual))

    input_ids = [PAD_ID] * pad_length + input_ids
    input_mask = [0] * pad_length + input_mask
    segment_ids = [3] * pad_length + segment_ids
    file_obj.writelines(str(tokens) + '\n')
    file_obj.writelines(str(input_ids) + '\n')
    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained(model)
    elif model == "xlnet-base-cased":
        return XLNetTokenizer.from_pretrained(model)
    elif model == "roberta-base":
        return RobertaTokenizer.from_pretrained(model)
    elif model == "roberta-large":
        return RobertaTokenizer.from_pretrained(model)
    else:
        raise ValueError(
            "Expected 'bert-base-uncased' or 'xlnet-base-cased, but received {}".format(
                model
            )
        )


def get_appropriate_dataset(data):

    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    #('num:', len(features))
    
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor(
        [f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)
    global data_index
    global data_no
    global word_no
    f_data_no = []
    for f in features:
        f_data_no.append(data_index)
        data_no[data_index] = f.data_no
        word_no[data_index] = f.sentence
        data_index += 1
    all_data_index = torch.tensor(f_data_no, dtype=torch.float)
    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
        all_data_index,
    )
    return dataset


def set_up_data_loader():
    print('start loader data...')
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)
    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]
    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )
    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps: int):
    
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob, backbone_layer=BL
    )

    if args.model == "bert-base-uncased":
        model = SPAMN_BertForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=1,
        )
    elif args.model == "xlnet-base-cased":
        model = SPAMN_XLNetForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=1
        )
    elif args.model == "roberta-base":
        model = SPAMN_RobertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=multimodal_config, num_labels=1
    )
    elif args.model == "roberta-large":
        model = SPAMN_RobertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=multimodal_config, num_labels=1
    )
    print('get model finished!')
    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_optimization_steps,
        num_training_steps=args.warmup_proportion * num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    #print('start train model...')
    
    model.train()
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.2fM" % (total/1e6))
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids, data_index = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        outputs = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask, labels=None,)
        # print(50*'#')
        # total = sum([param.nelement() for param in model.parameters()])
        # print("Number of parameter: %.2fM" % (total/1e6))
        # from thop import profile
        # flops, params = profile(model, inputs=(input_ids, visual, acoustic, segment_ids, input_mask, None,))
        # print("%s | %.2f | %.2f" % ('a', params / (1000 ** 2), flops / (1000 ** 3)))
        logits = outputs[0]
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))
        #np.savetxt('bn1.txt', bn1.cpu().detach().numpy())
        #np.savetxt('bn2.txt', bn2.cpu().detach().numpy())
    
        #print(loss, m)
        #torch.norm(m[0] - m[1]) + BETA * (m[0] - m[2]) + GAMA * torch.norm(m[1] - m[2])

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            '''
            for name,parameters in model.named_parameters():
                if name == 'bert.SPAMN.WL':
                    print(name,':',parameters[0], parameters[1], parameters.requires_grad, parameters.grad)
            else:
                'no canshu'
            '''
    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, optimizer):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids, data_index = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]
            
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []  
    label_indexs = []
    lv_attn_weights = [] 

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids, data_index = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )

            logits = outputs[0]
            #print('logitshape:', logits.shape)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            data_index = data_index.detach().cpu().numpy()
            #lv_attn_weight = m.detach().cpu().numpy()
            temp = []
            #temp = [lv_attn_weight[i] for i in range(len(lv_attn_weight))]

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)
            label_indexs.extend(data_index)
            lv_attn_weights.extend(temp)
        
        #for i in range(len(lv_attn_weights)):
        #    attn_map(str(i), lv_attn_weights[i])
        #    print(i, data_index[i], data_no[data_index[i]], word_no[data_index[i]])

        preds = np.array(preds)
        labels = np.array(labels)
        label_indexs = np.array(label_indexs)



    return preds, labels, label_indexs, lv_attn_weights


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False, epoch=-1, test=False):

    preds, y_test, label_indexs, m = test_epoch(model, test_dataloader)
    #if epoch == 35:
    #    save_model(model, name='hmia_xlnet_mosei_35_1')
    #des = np.array(['245582', '136.421', '140.684'])
    s_preds = preds
    s_y_test = y_test
    # if epoch == 50:
    #     print(len(m), len(preds))
    #     for i in range(len(preds)):
    #             #print(data_no[label_indexs[i]][0], type(data_no[label_indexs[i]][0]))
    #         print(data_no[label_indexs[i]][0])
    #         #if (data_no[label_indexs[i]][0] == des).all():
    #         #print(i, preds[i], y_test[i],label_indexs[i], data_no[label_indexs[i]], word_no[label_indexs[i]])
    #             #content = f'{i}, {preds[i]}, {y_test[i]}, {label_indexs[i]}, {data_no[label_indexs[i]]}, {word_no[label_indexs[i]]}'
    #         np.savetxt('attnweight/' + str(data_no[label_indexs[i]][0]) + '.txt', m[i])
    #             #attn_map(str(0), lv_attn_weights[i])
    #             #
    

    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)
        #
    #if (acc > args.best_acc) and (test is False):
        #name = args.model + '_' + str(args.seed) + '_'+ os.environ["CUDA_VISIBLE_DEVICES"]
        #print(f'{args.best_acc}->{acc} save model {name} finished!')
        #save_model(model, name=name)
        #args.best_acc = acc
    
    #save_model(model, name= str(args.seed) + '_'+ os.environ["CUDA_VISIBLE_DEVICES"])
    file_name = str(args.seed) + '_'+ os.environ["CUDA_VISIBLE_DEVICES"]
    file_obj = open(f'case/{epoch}_{file_name}.txt', 'w')
    for i in range(len(preds)):
        content = f'{i}, {s_preds[i]}, {s_y_test[i]}, {label_indexs[i]}, {data_no[label_indexs[i]]}, {word_no[label_indexs[i]]}'
        file_obj.writelines(content + '\n')
            #attn_map(str(i), lv_attn_weights[i].transpose(1, 0))
    file_obj.close()
    
    return acc, mae, corr, f_score


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
):
    valid_losses = []
    test_accuracies = []
    mae = []
    cc = []

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        test_acc, test_mae, test_corr, test_f_score = test_score_model(model, test_data_loader, epoch=epoch_i)

        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_acc:{}".format(
                epoch_i, train_loss, valid_loss, test_acc
            )
        )

        valid_losses.append(valid_loss)
        test_accuracies.append(test_acc)
        mae.append(test_mae)
        cc.append(test_corr)
        # wandb.log(
        #     (
        #         {
        #             "best_test_acc":max(test_accuracies),
        #             "best_valid_loss": min(valid_losses),
        #             "best_test_corr": max(cc),
        #             "best_test_mae": min(mae),
        #             "train_loss": train_loss,
        #             "valid_loss": valid_loss,
        #             "test_acc": test_acc,
        #             "test_mae": test_mae,
        #             "test_corr": test_corr,
        #             "test_f_score": test_f_score,
        #         }
        #     )
        # )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # wandb.init(project="Roberta_MOSI")
    # wandb.config.update(args)
    #args.seed = 8690
    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    # for step, batch in enumerate(tqdm(train_data_loader, desc="Iteration")):
    #     #batch = tuple(t.to(DEVICE) for t in batch)
    #     input_ids, visual, acoustic, input_mask, segment_ids, label_ids, data_index = batch
    #     print('label:', step, input_ids.shape, label_ids[0])
    #     print(data_index)
    #     print(type(data_index))
    #     print(data_index.shape)
    #     print(data_no[int(data_index[0])])
    #     exit(0)
    
    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    #print('start train...')
    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
