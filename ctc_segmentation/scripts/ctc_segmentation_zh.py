# -*- coding: utf-8 -*-
# @Time  : 2021/7/13 13:38
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : ctc_segmentation_zh.py
import json
import ctc_segmentation
import soundfile
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "zh-CN"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
DEVICE = "cpu"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.to(DEVICE)

vocab_dict = []


def segmenetation(wav_file):
    speech_array, sampling_rate = soundfile.read(wav_file)

    assert sampling_rate == 16000

    features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
    input_values = features.input_values
    attention_mask = features.attention_mask

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    # Split the transcription into words and handle as separate utterances

    transcript_file = wav_file.replace(".wav", "_output.txt")
    with open(transcript_file, "r", encoding='utf-8') as f:
        text = f.readlines()
        text = [t.strip() for t in text if t.strip()]

    # CTC log posteriors inference
    with torch.no_grad():
        softmax = torch.nn.LogSoftmax(dim=-1)
        lpz = softmax(logits)[0].cpu().numpy()

    index_duration = speech_array.shape[0] / lpz.shape[0] / sampling_rate

    # CTC segmentation preparation
    with open('F:\pythonProject\SpeechTextDatasetConstruct\\asr\wav2vec\\vocab.json', 'r', encoding='utf-8') as fr:
        content = fr.read()
        char_list = [key for key, value in json.loads(content).items()]

    # char_list = [x.lower() for x in vocab_dict.keys()]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = index_duration
    config.min_window_size = 6400
    # config.score_min_mean_over_L = 10
    # CTC segmentation
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, text)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, lpz, ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
    # print segments
    for word, segment in zip(text, segments):
        print(f"{segment[0]:.2f} {segment[1]:.2f} {segment[2]:3.4f} {word}")


if __name__ == '__main__':
    segmenetation('F:\pythonProject\SpeechTextDatasetConstruct\data\学习强国.wav')