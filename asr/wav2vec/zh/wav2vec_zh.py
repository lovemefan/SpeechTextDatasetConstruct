# -*- coding: utf-8 -*-
# @Time  : 2021/7/13 13:40
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : wav2vec_zh.py

import re

import soundfile
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "zh-CN"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
DEVICE = "cpu"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.to(DEVICE)


def recognized(wav_path):
    """
    :param wav_path:
    :return:
    """

    speech_array, sampling_rate = soundfile.read(wav_path)

    assert sampling_rate == 16000

    features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
    input_values = features.input_values
    attention_mask = features.attention_mask

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    # Split the transcription into words and handle as separate utterances
    text = transcription.split()
    # CTC log posteriors inference
    with torch.no_grad():
        softmax = torch.nn.LogSoftmax(dim=-1)
        lpz = softmax(logits)[0].cpu().numpy()

    duration = speech_array.shape[0] / lpz.shape[0] / sampling_rate
    return text[0], lpz, duration


if __name__ == '__main__':
    text, _ = recognized('/data/ylylbs-001.wav_3.250-6.800.wav')
    print(text)
