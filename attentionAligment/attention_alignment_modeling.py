# -*- coding: utf-8 -*-
# @Time  : 2022/4/23 16:10
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : attention_alignment_modeling.py
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers.file_utils import ModelOutput

from models.AttentionAlignment import Wav2Vec2ForAttentionAlignment

import numpy as np

import jiwer
import re
import argparse
import soundfile as sf
import librosa
import torch
from itertools import chain, groupby
import json


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

from datasets import load_dataset, load_metric, load_from_disk
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Trainer, TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from transformers import BertConfig


def get_phone_ids(phones):
    '''
    convert phone sequence to ids
    '''
    ids = []
    punctuation = set('.,!?')
    for p in phones:
        if re.match(r'^\w+?|\|$', p):
            ids.append(mapping_phone2id.get(p, mapping_phone2id['<unk>']))
        elif p in punctuation:
            ids.append(mapping_phone2id.get('<sil>'))
    ids = [0] + ids
    if ids[-1] != 0:
        ids.append(0)
    return ids  # append silence token at the beginning


def audio_preprocess(path):
    if SAMPLING_RATE == 32000:
        features, _ = librosa.core.load(path, sr=32000)
    else:
        features, _ = sf.read(path)
    return processor(features, sampling_rate=16000, return_tensors='pt').input_values.squeeze()


def seq2duration(phones, resolution=0.02):
    counter = 0
    out = []
    for p, group in groupby(phones):
        length = len(list(group))
        out.append((round(counter * resolution, 2), round((counter + length) * resolution, 2), p))
        counter += length
    return out


id_phoneme_pitch_map = {1: 5, 2: 8, 3: 33, 4: 18, 5: 6, 6: 12}
phoneme_pitch_id_map = {v: k for k, v in id_phoneme_pitch_map.items()}


def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    phonemes = []
    batch['input_values'] = batch['path']
    del batch['path']
    for char in batch['phoneme']:
        if char != ' ' or char == "|":
            phonemes.append(char)
    batch['labels'] = phonemes
    if args.phoneme_embedding:
        phoneme_pitch_ids = np.ones(len(batch['labels']))
        for j in range(len(batch['labels'])):
            value = batch['labels'][j]
            if value in id_phoneme_pitch_map.values():
                phoneme_pitch_ids[begin:end] *= phoneme_pitch_id_map[value]
                begin = j
            else:
                end = j + 1
        batch['phoneme_pitch_ids'] = phoneme_pitch_ids
    return batch


@dataclass
class SpeechCollatorWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        # get phone features
        label_features = [{"input_ids": get_phone_ids(feature["labels"])} for feature in features]
        text_len = [len(i['input_ids']) for i in label_features]

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_attention_mask=self.return_attention_mask,
                return_tensors="pt",
            )

            # get speech features
        input_features = [{"input_values": audio_preprocess(feature["input_values"])} for feature in features]
        mel_len = [model._get_feat_extract_output_lengths(len(i['input_values'])) for i in input_features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        batch_size, raw_sequence_length = batch['input_values'].shape
        sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)

        mask_prob = torch.randint(size=(1,), low=1, high=40) / 100
        batch['mask_time_indices'] = torch.BoolTensor(
            _compute_mask_indices((batch_size, sequence_length), mask_prob=mask_prob, mask_length=2))
        batch['frame_len'] = torch.tensor(mel_len)
        batch["text_len"] = torch.tensor(text_len)
        batch['labels'] = labels_batch["input_ids"]  # .masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch['labels_attention_mask'] = labels_batch['attention_mask']
        if args.phoneme_embedding:
            phoneme_pitch_ids = torch.zeros_like(labels_batch['input_ids'])
            phoneme_pitchs = [{"phoneme_pitch_ids": feature["phoneme_pitch_ids"][:self.max_length_labels]} for
                              feature in features]
            for index, item in enumerate(phoneme_pitchs):
                phoneme_pitch_ids[index, :len(item['phoneme_pitch_ids'])] = torch.Tensor(item['phoneme_pitch_ids'])

            batch["phoneme_pitch_ids"] = phoneme_pitch_ids
        else:
            batch["phoneme_pitch_ids"] = None
        # try for oov
        torch.cuda.empty_cache()
        return batch


def compute_metrics(pred):
    def decode(ids):
        results = []
        for index in range(ids.shape[0]):
            result = "".join([mapping_id2phone.get(i, '<unk>') for i in ids[index] if i != 0 and i !=1 and i !=-100])
            result = re.sub(r"(.*?)\1+", r"\1", result)
            results.append(result)
        return results

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_text = decode(pred_ids)
    # print(pred_text)
    label = decode(pred.label_ids)
    # print(label)
    wer = 0
    cer = 0
    for item in zip(pred_text, label):
        try:
            wer += jiwer.wer(item[0].replace('|', ' '), item[1].replace('|', ' '))
            cer += jiwer.cer(item[0].replace('|', ' '), item[1].replace('|', ' '))
        except ValueError:
            print(item[0], item[1])
    wer = wer/len(label)
    return {"phone_wer": wer, "phone_cer": cer}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,
                        default='/root/dataset/speechDataset/Vietnamese/mainfest_for_attention_alignment/train.tsv')
    # parser.add_argument('--wav2vec2', type=str, default="/root/data/dataset/wav2vec_of_vi_for_transformers")
    parser.add_argument('--wav2vec2', type=str, required=True, default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument('--bert', type=str, default="/root/data/dataset/text/vietnamese/bert-phones/checkpoint-13000")
    parser.add_argument('--out_dir', type=str, default="./checkpoint/wav2vec2-attention-align")
    parser.add_argument('--restore_from_checkpoint', type=str, default=None)
    parser.add_argument('--phoneme_embedding', action="store_true")
    parser.add_argument('--vocab', type=str, required=True)

    # parser.add_argument('--train_data', type=str,
    #                     default='/userdata/zlf/SpeechTextDatasetConstruct/mainfest_for_attention_alignment/train.tsv')
    # # parser.add_argument('--wav2vec2', type=str, default="/userdata/zlf/pretrain-models/transformers/vi-phoneme")
    # parser.add_argument('--wav2vec2', type=str, default="facebook/wav2vec2-large-xlsr-53")
    # parser.add_argument('--bert', type=str, default="/userdata/zlf/pretrain-models/transformers/bert-phones/checkpoint-13000")
    # parser.add_argument('--out_dir', type=str, default="./models/wav2vec2-attention-align")
    #
    parser.add_argument('--val_data', default="/root/dataset/speechDataset/Vietnamese/mainfest_for_attention_alignment/dev.tsv", type=str)
    parser.add_argument('--test_data', default=None, type=str)
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--mix_attention', action="store_true")
    parser.add_argument('--ignore_key', type=list, default=['attentions', 'diversity_loss', 'ctc_loss', 'contrastive_loss', 'align_loss'])

    parser.add_argument('--sampling_rate', type=float, default=16000)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()
    print(args)
    '''
        Load dataset
        '''
    corpus = load_dataset("csv", delimiter='\t', data_files=[args.train_data])
    val_corpus = load_dataset("csv",  delimiter='\t', data_files=[args.val_data])
    corpus = corpus.map(prepare_dataset)
    val_corpus = val_corpus.map(prepare_dataset)


    WEIGHT = args.weight
    SAMPLING_RATE = args.sampling_rate


    mapping_phone2id = json.load(open(args.vocab, 'r'))
    mapping_id2phone = {v: k for k, v in mapping_phone2id.items()}

    tokenizer = Wav2Vec2CTCTokenizer(args.vocab, unk_token="<unk>", pad_token="<pad>", word_delimiter_token="")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    data_collator = SpeechCollatorWithPadding(processor=processor)

    config = Wav2Vec2Config.from_pretrained(args.wav2vec2)
    bert_config = BertConfig.from_pretrained(args.bert)
    config.bert_config = bert_config
    config.pad_token_id = tokenizer.pad_token_id
    ctc_vocab_size = 156
    config.vocab_size = ctc_vocab_size
    config.mix_attention = args.mix_attention
    config.withPhonemePitchEmbedding = args.phoneme_embedding
    logging.info(f"mix_attention: {args.mix_attention}")
    config.ctc_loss_reduction = 'sum'
    model = Wav2Vec2ForAttentionAlignment(config, args)

    if args.restore_from_checkpoint is None:
        model.initialize_phone_model(args.bert)
        model.initialize_wav2vec2_model(args.wav2vec2)

    model.freeze_feature_extractor()
    model.bert.freeze_feature_extractor()
    model.config.bert_config = None
    total = sum([param.nelement() for param in model.parameters()])

    print(model)
    print("Number of model parameter: %.2fM" % (total / 1e6))

    # training settings
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        group_by_length=True,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=16,
        evaluation_strategy="steps",
        num_train_epochs=10,
        fp16=False,
        save_steps=100,
        eval_steps=100,
        logging_steps=100,
        learning_rate=args.lr,
        weight_decay=0.0001,
        warmup_steps=500,
        save_total_limit=100,
        ignore_data_skip=True,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=corpus['train'],
        eval_dataset=val_corpus['train'],
        tokenizer=processor.feature_extractor,
    )
    if args.restore_from_checkpoint:
        trainer.train(args.restore_from_checkpoint, ignore_keys_for_eval=args.ignore_key)
    else:
        trainer.train(ignore_keys_for_eval=args.ignore_key)
