# -*- coding: utf-8 -*-
# @Time  : 2022/4/8 14:19
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : AttentionAligment.py
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from transformers import Wav2Vec2ForCTC, Wav2Vec2ForPreTraining
from transformers.file_utils import ModelOutput

from attentionAligment.modules.MixedChunkAttention import MixedChunkAttention
from transformers.models.wav2vec2.modeling_wav2vec2 import  Wav2Vec2ForCTC
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from attentionAligment.models.PhonemeEncoder import BertForMaskedPhoneLM


class ForwardSumLoss(torch.nn.Module):
    '''
    Implementation from: https://nv-adlr.github.io/one-tts-alignment
    '''

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

@dataclass
class AttentionAlignOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None
    ctc_loss: Optional[torch.FloatTensor] = None
    align_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ForAttentionAlignment(Wav2Vec2ForPreTraining):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.bert = BertForMaskedPhoneLM(config.bert_config)

        # make sure that project_hid & project_q are initialized like normal linear layers
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
        self.lm_head = nn.Linear(768, config.vocab_size)
        self.project_hid = nn.Linear(768*2, config.proj_codevector_dim)
        self.dropout = nn.Dropout(config.final_dropout)
        if config.mix_attention:
            self.attention = MixedChunkAttention(dim=768)
        else:
            self.attention = CrossAttention(768)
        self.align_loss = ForwardSumLoss()

    def freeze_wav2vec2(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def initialize_phone_model(self, path):

        self.bert = BertForMaskedPhoneLM.from_pretrained(path)
        self.config.bert_config = None
        self.bert.freeze_feature_extractor()

    def initialize_wav2vec2_model(self, path):
        weights = Wav2Vec2ForCTC.from_pretrained(path).state_dict()
        # del weights['lm_head.weight']
        # del weights['lm_head.bias']
        state_dict = self.state_dict()
        weights = {k: v for k, v in weights.items() if k in state_dict.keys()}
        state_dict.update(weights)

        self.load_state_dict(state_dict)
        # if self.args.sampling_rate != 32000:
        #     self.wav2vec2.feature_extractor.conv_layers[6].conv.stride = (1,)
        #     self.config.conv_stride[-1] = 1
        self.freeze_feature_extractor()


        # self.config.num_negatives = 200
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
            frame_len=None,
            phoneme_pitch_ids=None
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

        # phone embeddings
        if phoneme_pitch_ids is None:
            phone_hidden = self.bert(input_ids=labels, attention_mask=labels_attention_mask, phoneme_pitch_ids=phoneme_pitch_ids).hidden_states[-1]
        else:
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
        ctc_logits = self.lm_head(hidden_states)
        # ctc_loss doesn't support fp16
        log_probs = nn.functional.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        input_lengths = torch.sum(attention_mask, dim=1).long()
        target_lengths = torch.sum(labels_attention_mask, dim=1).long()
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = nn.functional.ctc_loss(
                log_probs,
                labels.masked_select(labels_attention_mask==1),
                input_lengths,
                target_lengths,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )
        loss = None

        # try for oov
        torch.cuda.empty_cache()
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

        # expanded_attention_mask = attention_mask.unsqueeze(2).repeat(1,1,energy.size(2)) * labels_attention_mask.unsqueeze(1).repeat(1,energy.size(1),1)
        # expanded_attention_mask = (1-expanded_attention_mask)*-10000.0
        # phone_attention = torch.softmax((energy+expanded_attention_mask).transpose(2,1),dim=-1)
        # phone_emb = torch.bmm(phone_attention,frame_hidden)
        # prediction_scores = self.phone_rnn(phone_emb,text_len)
        # labels = labels.masked_fill(labels_attention_mask.ne(1), -100)
        # inter_phone = F.cosine_similarity(phone_emb[:,:-1,:],phone_emb[:,1:,:],dim=-1)*labels_attention_mask[:,1:]
        # interphone_loss = torch.sum(inter_phone)/torch.sum(labels_attention_mask[:,1:])

        loss = contrastive_loss + self.args.weight * align_loss + ctc_loss + diversity_loss # + interphone_loss
        # loss = align_loss + ctc_loss # + interphone_loss

        # try for oov
        torch.cuda.empty_cache()
        return AttentionAlignOutput(
            loss=loss,
            contrastive_loss=contrastive_loss,
            align_loss=align_loss,
            ctc_loss=ctc_loss,
            diversity_loss=diversity_loss,
            logits=ctc_logits,
            hidden_states=outputs.hidden_states,
            attentions=energy
        )


class ConvBank(nn.Module):
    '''
    Implementation from: https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/libri_phone/model.py
    '''

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


class RNN(nn.Module):

    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, num_layers=1, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, out_dim))

    def forward(self, embeddings, lens):
        packed_input = pack_padded_sequence(embeddings, lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.linear(out)
        return out


class CrossAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        #        self.v = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, q, k, v, mask):
        frame_hidden = self.q(q)
        phone_hidden = self.k(k)

        energy = torch.bmm(frame_hidden, phone_hidden.transpose(2, 1))
        attention_mask = (1 - mask) * -10000.0
        energy = energy + attention_mask.unsqueeze(1).repeat(1, energy.size(1), 1)

        att_matrix = torch.softmax(energy, dim=-1)
        att_out = torch.bmm(att_matrix, k)
        att_out = torch.cat([att_out, frame_hidden], dim=-1)
        #        att_out = self.layer_norm(att_out + frame_hidden)

        return att_out, energy
