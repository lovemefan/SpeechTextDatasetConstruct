# -*- coding: utf-8 -*-
# @Time  : 2022/3/31 12:11
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : PhonemeEncoder.py
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertAttention, BertLayer, BertEncoder

from attentionAligment.modules.GateAttentionUnit import GAUForTransformers


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


class BertAttentionWithGAU(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = GAUForTransformers(config)

    def prune_heads(self, heads):
        pass

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            mask=head_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLartWithGAU(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithGAU(config)


class BertEncoderWithGAU(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([BertLartWithGAU(config) for _ in range(config.num_hidden_layers)])


class BertForMaskedPhoneLM(BertForMaskedLM):

    def __init__(self, config):
        super().__init__(config)
        self.cnn = ConvBank(config.hidden_size,
                            config.hidden_size,
                            config.convbank,
                            config.hidden_size,
                            config.hidden_size,
                            config.hidden_dropout_prob)
        self.out_linear = nn.Linear(config.hidden_size, config.vocab_size)
        if config.withGAU:
            self.bert.encoder = BertEncoderWithGAU(config)

    def freeze_feature_extractor(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=True,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        prediction_scores = self.cnn(outputs.hidden_states[-1])
        prediction_scores = self.out_linear(prediction_scores)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

