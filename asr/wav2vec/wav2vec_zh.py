# -*- coding: utf-8 -*-
# @Time  : 2021/7/13 13:40
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : wav2vec_zh.py

import re

import soundfile
import torch, transformers, ctc_segmentation
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

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    # Split the transcription into words and handle as separate utterances
    text = transcription.split()
    # CTC log posteriors inference
    with torch.no_grad():
        softmax = torch.nn.LogSoftmax(dim=-1)
        lpz = softmax(logits)[0].cpu().numpy()

    return text, lpz


if __name__ == '__main__':
    text, _ = recognized('F:\pythonProject\SpeechTextDatasetConstruct\data\《摸金天师》第001章_百辟刀.wav')
    print(text)
