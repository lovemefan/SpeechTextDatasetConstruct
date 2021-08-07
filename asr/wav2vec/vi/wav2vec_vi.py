# -*- coding: utf-8 -*-
# @Time  : 2021/8/6 21:15
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : wav2vec_viu.py
from pathlib import Path

import soundfile
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


model_path = Path('F:\pythonProject\SpeechTextDatasetConstruct\\asr\wav2vec\\vi\\vi_checkpoint_wer0.16')
model = Wav2Vec2ForCTC.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(model_path)


def recognized(wav_path):
    """
    :param wav_path:
    :return:
    """
    speech_array, sampling_rate = soundfile.read(wav_path)
    assert sampling_rate == 16000
    input_dict = processor(speech_array, return_tensors="pt", padding=True)

    # logits = model(input_dict.input_values.to("cuda")).logits
    logits = model(input_dict.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    # Split the transcription into words and handle as separate utterances
    # CTC log posteriors inference
    with torch.no_grad():
        softmax = torch.nn.LogSoftmax(dim=-1)
        lpz = softmax(logits)[0].cpu().numpy()

    duration = speech_array.shape[0] / lpz.shape[0] / sampling_rate
    return transcription, lpz, duration

if __name__ == '__main__':
    text, _, _1 = recognized('F:\pythonProject\SpeechTextDatasetConstruct\data\\VIVOSDEV02_R122.wav')
    print(text)
