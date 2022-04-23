# -*- coding: utf-8 -*-
# @Time  : 2022/4/23 16:10
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : attention_alignment_modeling.py
import re
import argparse
from pathlib import Path
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from itertools import chain, groupby
import json


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from g2p_en import G2p
from datasets import load_dataset, load_metric, load_from_disk
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Trainer, TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, Wav2Vec2ForPreTrainingOutput, \
    Wav2Vec2ForCTC
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
        hidden = F.dropout(F.relu(self.in_linear(features)), p=self.drop_p)

        conv_feats = []
        hidden = hidden.transpose(1, 2).contiguous()
        for cnn in self.cnns:
            conv_feats.append(cnn(hidden))
        hidden = torch.cat(conv_feats, dim=1).transpose(1, 2).contiguous()
        hidden = F.dropout(F.relu(hidden), p=self.drop_p)

        predicted = self.out_linear(hidden)
        return predicted


class Wav2Vec2ForAttentionAlignment(Wav2Vec2ForPreTraining):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertForMaskedPhoneLM(config.bert_config)
        self.cnn = ConvBank(config.hidden_size, 384, [1], 384, 384, 0.1)
        # self.lm_head = nn.Linear(config.hidden_size,config.vocab_size)
        #        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        # self.phone_rnn = RNN(384, config.vocab_size)

        self.attention = MixedChunkAttention(384)
        self.align_loss = ForwardSumLoss()

    def freeze_wav2vec2(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def initialize_phone_model(self, path):

        self.bert = BertForMaskedPhoneLM.from_pretrained(path)

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
        att_out, energy = self.attention(frame_hidden, phone_hidden, labels_attention_mask)

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

        #        loss_fct = nn.CrossEntropyLoss()

        #        phone_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        loss = None
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
            # num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            # diversity_loss = (num_codevectors - codevector_perplexity) / num_codevectors

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

            loss = contrastive_loss + WEIGHT * align_loss  # + interphone_loss

        return CausalLMOutput(
            loss=loss, logits=transformer_features, hidden_states=outputs.hidden_states, attentions=energy
        )


def get_phones(sen):
    '''
    convert texts to phone sequence
    '''
    sen = g2p(sen)
    sen = [re.sub(r'\d', '', p) for p in sen]
    return sen


def get_phone_ids(phones):
    '''
    convert phone sequence to ids
    '''
    ids = []
    punctuation = set('.,!?')
    for p in phones:
        if re.match(r'^\w+?$', p):
            ids.append(mapping_phone2id.get(p, mapping_phone2id['[UNK]']))
        elif p in punctuation:
            ids.append(mapping_phone2id.get('[SIL]'))
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


def prepare_common_voice_dataset(batch):
    # check that all files have the correct sampling rate

    batch["input_values"] = re.search(r'(.*?)\.mp3', batch['path']).group(1) + '.wav'
    # batch["input_values"] = batch['path']
    # batch['labels'] = batch['sentence']
    batch['labels'] = "Lorem ipsum dolor sit amet, consectetur adipisicing elit"
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
        label_features = [{"input_ids": get_phone_ids(get_phones(feature["labels"]))} for feature in features]
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

        return batch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,
                        default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/asr/common_voice_filtered')
    parser.add_argument('--val_data', default=None, type=str)
    parser.add_argument('--test_data', default=None, type=str)
    parser.add_argument('--out_dir', type=str, default="./models/wav2vec2-base-cv-attention-align-10ms-1.0")
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--sampling_rate', type=float, default=16000)

    args = parser.parse_args()

    WEIGHT = args.weight
    SAMPLING_RATE = args.sampling_rate

    g2p = G2p()
    mapping_phone2id = json.load(open("vocab-ctc.json", 'r'))
    mapping_id2phone = {v: k for k, v in mapping_phone2id.items()}

    tokenizer = Wav2Vec2CTCTokenizer("vocab-ctc.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base')
    bert_config = BertConfig.from_pretrained(Path('/userdata/zlf/charsiu/bert-phoneme'))
    config.bert_config = bert_config
    config.pad_token_id = tokenizer.pad_token_id
    config.vocab_size = len(tokenizer)
    config.ctc_loss_reduction = 'mean'
    model = Wav2Vec2ForAttentionAlignment(config)
    # model.freeze_feature_extractor()
    #    model.initialize_phone_model('../bert-phone')

    weights = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base').state_dict()
    del weights['lm_head.bias']
    del weights['lm_head.weight']
    #     weights = torch.load('./models/neural_attention_aligner_forwardsum_10ms_true_quantizer.pt').state_dict()
    #     weights = model.from_pretrained("charsiu/en_w2v2_fs_10ms").state_dict()
    state_dict = model.state_dict()
    #    weights = {k:v for k,v in weights.items() if k in state_dict.keys()}
    state_dict.update(weights)

    model.load_state_dict(state_dict)
    if SAMPLING_RATE != 32000:
        model.wav2vec2.feature_extractor.conv_layers[6].conv.stride = (1,)
        model.config.conv_stride[-1] = 1
    model.freeze_feature_extractor()
    # model.bert.freeze_feature_extractor()
    model.config.bert_config = None

    model.config.num_negatives = 50
    for param in model.quantizer.parameters():
        param.requires_grad = False
    for param in model.project_q.parameters():
        param.requires_grad = False

    '''
    Load dataset
    '''
    # common_voice = load_from_disk('/dataset/speech/common_voice/cv-corpus-6.1-2020-12-11/en')
    common_voice = load_dataset('common_voice', 'as')
    common_voice = common_voice.map(prepare_common_voice_dataset)
    print(len(common_voice))
    print(common_voice['train'][0])

    data_collator = SpeechCollatorWithPadding(processor=processor)

    # training settings
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        group_by_length=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        #                                      evaluation_strategy="steps",
        num_train_epochs=1,
        fp16=False,
        save_steps=500,
        #                                      eval_steps=1000,
        logging_steps=500,
        learning_rate=3e-4,
        weight_decay=0.0001,
        warmup_steps=500,
        save_total_limit=2,
        ignore_data_skip=True,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        #                            compute_metrics=compute_metrics,
        train_dataset=common_voice['train'],
        #                            eval_dataset=libris_train_prepared,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
