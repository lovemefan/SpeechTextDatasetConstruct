# -*- coding: utf-8 -*-
# @Time  : 2021/7/13 13:38
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : ctc_segmentation_zh.py

def segmenetation():
    # CTC segmentation preparation
    char_list = [x.lower() for x in vocab_dict.keys()]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = speech_array.shape[0] / lpz.shape[0] / sampling_rate
    # CTC segmentation
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, text)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, lpz, ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
    # print segments
    for word, segment in zip(text, segments):
        print(f"{segment[0]:.2f} {segment[1]:.2f} {segment[2]:3.4f} {word}")