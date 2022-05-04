# -*- coding: utf-8 -*-
# @Time  : 2022/4/23 16:10
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : attention_alignment_modeling.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attentionAligment.models.AttentionAlignment import CrossAttention
import re
import argparse
from pathlib import Path
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices,  \
    Wav2Vec2ForCTC, Wav2Vec2GumbelVectorQuantizer
from transformers.modeling_outputs import CausalLMOutput, MaskedLMOutput
from transformers import BertForMaskedLM, BertConfig

from attentionAligment.models.PhonemeEncoder import BertForMaskedPhoneLM
from attentionAligment.modules.MixedChunkAttention import MixedChunkAttention


class ForwardSumLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(ForwardSumLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        #        self.off_diag_penalty = off_diag_penalty
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, text_lens, mel_lens):
        """
        Args:
        attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
        batched tensor of attention log
        probabilities, padded to length
        of longest sequence in each dimension
        text_lens: batch-D vector of length of
        each text sequence
        mel_lens: batch-D vector of length of
        each mel sequence
        """
        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        attn_logprob_pd = F.pad(input=attn_logprob,
                                pad=(1, 0, 0, 0, 0, 0, 0, 0),
                                value=self.blank_logprob)
        cost_total = 0.0
        # for-loop over batch because of variable-length
        # sequences
        for bid in range(attn_logprob.shape[0]):
            # construct the target sequence. Every
            # text token is mapped to a unique sequence number,
            # thereby ensuring the monotonicity constraint
            target_seq = torch.arange(1, text_lens[bid] + 1)
            target_seq = target_seq.unsqueeze(0)
            curr_logprob = attn_logprob_pd[bid].permute(1, 0, 2)
            curr_logprob = curr_logprob[:mel_lens[bid], :, :text_lens[bid] + 1]

            #            curr_logprob = curr_logprob + self.off_diagonal_loss(curr_logprob,text_lens[bid]+1,mel_lens[bid])
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            cost = self.CTCLoss(curr_logprob,
                                target_seq,
                                input_lengths=mel_lens[bid:bid + 1],
                                target_lengths=text_lens[bid:bid + 1])
            cost_total += cost
        # average cost over batch
        cost_total = cost_total / attn_logprob.shape[0]
        return cost_total

    def off_diagonal_prior(self, log_prob, N, T, g=0.2):
        n = torch.arange(N).to(log_prob.device)
        t = torch.arange(T).to(log_prob.device)
        t = t.unsqueeze(1).repeat(1, N)
        n = n.unsqueeze(0).repeat(T, 1)

        #        W = 1 - torch.exp(-(n/N - t/T)**2/(2*g**2))

        #        penalty = log_prob*W.unsqueeze(1)
        #        return torch.mean(penalty)
        W = torch.exp(-(n / N - t / T) ** 2 / (2 * g ** 2))

        return torch.log_softmax(W.unsqueeze(1), dim=-1)


class ConvBank(nn.Module):
    def __init__(self, input_dim, output_class_num, kernels, cnn_size, hidden_size, dropout, **kwargs):
        super(ConvBank, self).__init__()
        self.drop_p = dropout

        self.in_linear = nn.Linear(input_dim, hidden_size)
        latest_size = hidden_size

        # conv bank
        self.cnns = nn.ModuleList()
        assert len(kernels) > 0
        for kernel in kernels:
            self.cnns.append(nn.Conv1d(latest_size, cnn_size, kernel, padding=kernel // 2))
        latest_size = cnn_size * len(kernels)

        self.out_linear = nn.Linear(latest_size, output_class_num)

    def forward(self, features):
        hidden = F.dropout(F.relu(self.in_linear(features), inplace=False), p=self.drop_p)

        conv_feats = []
        hidden = hidden.transpose(1, 2).contiguous()
        for cnn in self.cnns:
            conv_feats.append(cnn(hidden))
        hidden = torch.cat(conv_feats, dim=1).transpose(1, 2).contiguous()
        hidden = F.dropout(F.relu(hidden, inplace=False), p=self.drop_p)

        predicted = self.out_linear(hidden)
        return predicted


class Wav2Vec2ForAttentionAlignment(Wav2Vec2ForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertForMaskedPhoneLM(config.bert_config)
        self.cnn = ConvBank(config.hidden_size, 384, [1], 384, 384, 0.1)
        # self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)
        # self.lm_head = nn.Linear(config.hidden_size,config.vocab_size)
        # self.project_hid = nn.Linear(512, config.proj_codevector_dim)
        # self.phone_rnn = RNN(384, config.vocab_size)

        # make sure that project_hid & project_q are initialized like normal linear layers
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
        self.project_hid = nn.Linear(512, config.proj_codevector_dim)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(384, config.vocab_size)
        if config.mix_attention:
            self.attention = MixedChunkAttention(dim=384)
        else:
            self.attention = CrossAttention(384)
        self.align_loss = ForwardSumLoss()

    def freeze_wav2vec2(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def initialize_phone_model(self, path):

        self.bert = BertForMaskedPhoneLM.from_pretrained(path)
        self.config.bert_config = None
        # self.bert.freeze_feature_extractor()

    def initialize_wav2vec2_model(self, path):
        weights = Wav2Vec2ForCTC.from_pretrained(path).state_dict()
        del weights['lm_head.weight']
        del weights['lm_head.bias']
        state_dict = self.state_dict()
        weights = {k: v for k, v in weights.items() if k in state_dict.keys()}
        state_dict.update(weights)

        self.load_state_dict(state_dict)
        if SAMPLING_RATE != 32000:
            self.wav2vec2.feature_extractor.conv_layers[6].conv.stride = (1,)
            self.config.conv_stride[-1] = 1
        self.freeze_feature_extractor()


        self.config.num_negatives = 50
        for param in self.quantizer.parameters():
            param.requires_grad = False
        for param in self.project_q.parameters():
            param.requires_grad = False

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    @staticmethod
    def _sample_negatives(
            features: torch.FloatTensor, num_negatives: int, attention_mask: Optional[torch.LongTensor] = None
    ):
        """
        Sample `num_negatives` vectors from feature vectors.
        """
        batch_size, sequence_length, hidden_size = features.shape
        if sequence_length <= 1:
            raise ValueError(
                f"`features should have `sequence_length` > 1, but are of shape (batch_size, sequence_length, hidden_size) = ({batch_size, sequence_length, hidden_size})."
            )

        features = features.view(-1, hidden_size)  # BTC => (BxT)C

        with torch.no_grad():
            # get `num_negatives` random vector indices from the same utterance
            sampled_negative_indices = []
            for batch_idx in range(batch_size):
                high = attention_mask[batch_idx].sum() - 1 if attention_mask is not None else sequence_length - 1
                sampled_indices_slice = torch.randint(
                    0, high, size=(num_negatives * sequence_length,), device=features.device
                )
                sampled_negative_indices.append(sampled_indices_slice)

            sampled_negative_indices = torch.stack(sampled_negative_indices)

            # generate indices of the positive vectors themselves, repeat them `num_negatives` times
            feature_indices = (
                torch.arange(sequence_length, device=features.device)[:, None]
                    .expand(sequence_length, num_negatives)
                    .flatten()
            )

            # avoid sampling the same positive vector, but keep the distribution uniform
            sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1

        # correct for batch size
        for batch_idx in range(1, batch_size):
            sampled_negative_indices[batch_idx] += batch_idx * sequence_length

        # take negative vectors from sampled indices
        sampled_negatives = features[sampled_negative_indices.view(-1)]
        sampled_negatives = sampled_negatives.view(batch_size, sequence_length, num_negatives, hidden_size).permute(
            2, 0, 1, 3
        )

        return sampled_negatives

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            mask_time_indices=None,
            return_dict=None,
            labels=None,
            labels_attention_mask=None,
            text_len=None,
            frame_len=None
    ):

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # acoustic embeddings
        frame_hidden = outputs[0]
        #        frame_hidden = self.dropout(frame_hidden)
        frame_hidden = self.cnn(frame_hidden)

        # phone embeddings
        phone_hidden = self.bert(input_ids=labels, attention_mask=labels_attention_mask).hidden_states[-1]

        # compute cross attention

        att_out, energy = self.attention(frame_hidden, phone_hidden, frame_hidden, mask=labels_attention_mask)

        # start masked modeling
        # 0. remove the blank symbol
        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(torch.tanh(att_out))

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])
        quantized_features, codevector_perplexity = self.quantizer(extract_features, mask_time_indices)
        quantized_features = self.project_q(quantized_features)

        # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)


        # 3. compute CTC loss
        hidden_states = self.dropout(frame_hidden)
        logits = self.lm_head(hidden_states)
        # ctc_loss doesn't support fp16
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        input_lengths = torch.ones(log_probs.size()[1]).long()*log_probs.size()[0]
        target_lengths = torch.sum(torch.ones_like(labels), dim=-1).long()
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = nn.functional.ctc_loss(
                log_probs,
                labels.view(-1),
                input_lengths,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )
        loss = None
        # try for oov
        torch.cuda.empty_cache()
        if self.training:
            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss

            negative_quantized_features = self._sample_negatives(
                quantized_features, self.config.num_negatives
            )

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)
            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            preds = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - attention_mask.long()) * -100).transpose(0, 1).flatten()
            contrastive_loss = nn.functional.cross_entropy(preds.float(), target, reduction="mean")

            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = (num_codevectors - codevector_perplexity) / num_codevectors

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            expanded_labels_attention_mask = (1 - labels_attention_mask) * -10000.0
            expanded_labels_attention_mask = expanded_labels_attention_mask.unsqueeze(1).repeat(1, energy.size(1), 1)
            att = torch.log_softmax(energy + expanded_labels_attention_mask, dim=-1)
            align_loss = self.align_loss(att.unsqueeze(1), text_len, frame_len)

            #            expanded_attention_mask = attention_mask.unsqueeze(2).repeat(1,1,energy.size(2)) * labels_attention_mask.unsqueeze(1).repeat(1,energy.size(1),1)
            #            expanded_attention_mask = (1-expanded_attention_mask)*-10000.0
            #            phone_attention = torch.softmax((energy+expanded_attention_mask).transpose(2,1),dim=-1)
            #            phone_emb = torch.bmm(phone_attention,frame_hidden)
            #            prediction_scores = self.phone_rnn(phone_emb,text_len)
            #            labels = labels.masked_fill(labels_attention_mask.ne(1), -100)
            #            inter_phone = F.cosine_similarity(phone_emb[:,:-1,:],phone_emb[:,1:,:],dim=-1)*labels_attention_mask[:,1:]
            #            interphone_loss = torch.sum(inter_phone)/torch.sum(labels_attention_mask[:,1:])

            loss = contrastive_loss + WEIGHT * align_loss + ctc_loss # + interphone_loss
            # loss = align_loss + ctc_loss # + interphone_loss
        # try for oov
        torch.cuda.empty_cache()
        return CausalLMOutput(
            loss=loss, logits=transformer_features, hidden_states=outputs.hidden_states, attentions=energy
        )




def get_phone_ids(phones):
    '''
    convert phone sequence to ids
    '''
    ids = []
    punctuation = set('.,!?')
    for p in phones:
        if re.match(r'^\w+?$', p):
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


def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    phonemes = []
    batch['input_values'] = batch['path']
    del batch['path']
    for char in batch['phoneme']:
        if char != ' ' and char != '|':
            phonemes.append(char)
    batch['labels'] = phonemes
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
        # try for oov
        torch.cuda.empty_cache()
        return batch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,
                        default='/root/dataset/speechDataset/Vietnamese/mainfest_for_attention_alignment/train.tsv')
    # parser.add_argument('--wav2vec2', type=str, default="/root/data/dataset/wav2vec_of_vi_for_transformers")
    parser.add_argument('--wav2vec2', type=str, default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument('--bert', type=str, default="/root/data/dataset/text/vietnamese/bert-phones/checkpoint-13000")
    parser.add_argument('--out_dir', type=str, default="./checkpoint/wav2vec2-attention-align")
    parser.add_argument('--restore_from_checkpoint', type=str, default=None)


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
    parser.add_argument('--sampling_rate', type=float, default=16000)

    args = parser.parse_args()

    '''
        Load dataset
        '''
    corpus = load_dataset("csv", delimiter='\t', data_files=[args.train_data])
    val_corpus = load_dataset("csv",  delimiter='\t', data_files=[args.val_data])
    corpus = corpus.map(prepare_dataset)
    val_corpus = val_corpus.map(prepare_dataset)




    WEIGHT = args.weight
    SAMPLING_RATE = args.sampling_rate


    mapping_phone2id = json.load(open("vocab.json", 'r'))
    mapping_id2phone = {v: k for k, v in mapping_phone2id.items()}

    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    data_collator = SpeechCollatorWithPadding(processor=processor)

    config = Wav2Vec2Config.from_pretrained(args.wav2vec2)
    bert_config = BertConfig.from_pretrained(Path(args.bert))
    config.bert_config = bert_config
    config.pad_token_id = tokenizer.pad_token_id
    config.vocab_size = len(tokenizer)
    config.mix_attention = True
    config.ctc_loss_reduction = 'mean'
    model = Wav2Vec2ForAttentionAlignment(config)
    model.freeze_feature_extractor()
    if not args.restore_from_checkpoint:
        model.initialize_phone_model(args.bert)
        model.initialize_wav2vec2_model(args.wav2vec2)
    else:
        # avoid TypeError: Object of type BertConfig is not JSON serializable
        model.config.bert_config = None
    total = sum([param.nelement() for param in model.parameters()])

    print(model)
    print("Number of model parameter: %.2fM" % (total / 1e6))

    # training settings
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        group_by_length=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        #                                      evaluation_strategy="steps",
        num_train_epochs=20,
        fp16=True,
        save_steps=500,
        #                                      eval_steps=1000,
        logging_steps=500,
        learning_rate=3e-5,
        weight_decay=0.0001,
        warmup_steps=500,
        save_total_limit=2,
        ignore_data_skip=True,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        # compute_metrics=compute_metrics,
        train_dataset=corpus['train'],
        eval_dataset=val_corpus['train'],
        tokenizer=processor.feature_extractor,
    )
    if args.restore_from_checkpoint:
        trainer.train(args.restore_from_checkpoint)
    else:
        trainer.train()
