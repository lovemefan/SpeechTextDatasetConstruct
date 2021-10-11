# -*- coding: utf-8 -*-
# @Time  : 2021/9/23 10:47
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : wenet_recognize.py

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import numpy as np
import torch
import torchaudio
import yaml
from torch.nn.utils.rnn import pad_sequence
from torchaudio.compliance import kaldi

from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint


def load_feature(file_path):
    """
    load audio feature using fbank
    :param file_path: file path
    :return:
    """
    waveform, sampling_rate = torchaudio.load(file_path)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.1,
        energy_floor=0.0,
        sample_frequency=sampling_rate)
    mat = mat.detach().numpy()

    # padding
    xs_lengths = torch.from_numpy(np.array([mat.shape[0]], dtype=np.int32))

    # pad_sequence will FAIL in case xs is empty
    if len(mat) > 0:
        xs_pad = pad_sequence([torch.from_numpy(x).float() for x in mat],
                              True, 0)
    else:
        xs_pad = torch.Tensor([mat])

    xs_pad = xs_pad.unsqueeze(0)
    # xs_lengths = xs_lengths.unsqueeze(0)

    return xs_pad, xs_lengths


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring'
                        ],
                        default='attention',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    args = parser.parse_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    raw_wav = configs['raw_wav']
    # Init dataset and data loader
    # Init dataset and data loader
    test_collate_conf = copy.deepcopy(configs['collate_conf'])
    test_collate_conf['spec_aug'] = False
    test_collate_conf['spec_sub'] = False
    test_collate_conf['feature_dither'] = False
    test_collate_conf['speed_perturb'] = False
    if raw_wav:
        test_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
        test_collate_conf['wav_distortion_conf']['wav_dither'] = 0.0
    test_collate_func = CollateFunc(**test_collate_conf, raw_wav=raw_wav)
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['batch_type'] = 'static'
    dataset_conf['sort'] = False

    # Init asr model from configs
    model = init_asr_model(configs)

    # Load dict
    char_dict = {}
    with open(args.dict, 'r', encoding='utf-8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()
    with torch.no_grad(), open(args.result_file, 'w', encoding='utf-8') as fout:

        feats, feats_lengths = load_feature("E:\Corpus\越南语\\vivos\\vivos\\test\waves\VIVOSDEV01\VIVOSDEV01_R002.wav")
        feats = feats.to(device)
        feats_lengths = feats_lengths.to(device)

        hyps = model.attention_rescoring(
            feats,
            feats_lengths,
            decoding_chunk_size=args.decoding_chunk_size,
            num_decoding_left_chunks=args.num_decoding_left_chunks,
            simulate_streaming=args.simulate_streaming)

        content = ''
        for w in hyps[0]:
            if w == eos:
                break
            content += f"{char_dict[w]} "
        logging.info('{}'.format(content))
        fout.write('{}\n'.format(content))

