# -*- coding: utf-8 -*-
# @Time  : 2021/8/7 12:56
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : ctc_segmentation.py
import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import soundfile

from asr.wav2vec.vi.wav2vec_vi import ViASR
from asr.wav2vec.zh.wav2vec_zh import ZhASR
from ctc_segmentation_construct.scripts.utils import *

parser = argparse.ArgumentParser(description="CTC Segmentation")
parser.add_argument("--output_dir", default='output', type=str, help='Path to output directory')
parser.add_argument(
    "--data",
    type=str,
    required=True,
    help='Path to directory with audio files and associated transcripts (same respective names only formats are '
    'different or path to wav file (transcript should have the same base name and be located in the same folder'
    'as the wav file.',
)
parser.add_argument('--vocab', type=str, required=True, help='vocab.json file')
parser.add_argument('--window_len', type=int, default=8000, help='Window size for ctc segmentation algorithm')
parser.add_argument('--no_parallel', action='store_true', help='Flag to disable parallel segmentation')
parser.add_argument('--sample_rate', type=int, default=16000, help='Sampling rate')
parser.add_argument(
    '--language', type=str, default='zh', help='language, zh , vi',
)
parser.add_argument('--debug', action='store_true', help='Flag to enable debugging messages')

logger = logging.getLogger('ctc_segmentation')  # use module name

if __name__ == '__main__':

    args = parser.parse_args()
    # setup logger
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'ctc_segmentation_{args.window_len}.log')
    level = 'DEBUG' if args.debug else 'INFO'

    logger = logging.getLogger('CTC')
    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(handlers=handlers, level=level)

    if args.language == 'zh':
        asr_model = ZhASR()
    elif args.language == 'vi':
        asr_model = ViASR()
    else:
        raise ValueError(
            f'language {args.language} not a valid language. '
            f'or choose from zh, vi'
        )

    data = Path(args.data)
    output_dir = Path(args.output_dir)

    if os.path.isdir(data):
        audio_paths = data.glob("*.wav")
        data_dir = data
    else:
        audio_paths = [Path(data)]
        data_dir = Path(os.path.dirname(data))

    all_log_probs = []
    all_transcript_file = []
    all_segment_file = []
    all_wav_paths = []
    segments_dir = os.path.join(args.output_dir, 'segments')
    os.makedirs(segments_dir, exist_ok=True)

    for path_audio in audio_paths:
        transcript_file = os.path.join(data_dir, path_audio.name.replace(".wav", "_output.txt"))
        segment_file = os.path.join(
            segments_dir, f"{args.window_len}_" + path_audio.name.replace(".wav", "_output_segments.txt")
        )
        try:
            signal, sample_rate = soundfile.read(path_audio)

            if sample_rate != int(args.sample_rate):
                raise ValueError(
                    f'Sampling rate of the audio file {path_audio} doesn\'t match --sample_rate={args.sample_rate}'
                )
        except ValueError:
            logging.error(
                f"{path_audio} should be a .wav mono file with the sampling rate used for the ASR model training"
                f"specified with {args.sample_rate}."
            )
            raise

        transcription, log_probs, duration = asr_model.recognized(path_audio)
        logging.info(f'Duration: {duration}s, file_name: {path_audio}')

        all_log_probs.append(log_probs)
        all_segment_file.append(str(segment_file))
        all_transcript_file.append(str(transcript_file))
        all_wav_paths.append(path_audio)

    del asr_model

    try:
        if not os.path.exists(args.vocab):
            raise ValueError(
                f'vocab.json file {args.vocab} doesn\'t exist, please check the path of vocab.json '
            )
        # load vocabulary
        with open('/asr/wav2vec/zh/vocab.json', 'r', encoding='utf-8') as fr:
            content = fr.read()
            vocabulary = [key for key, value in json.loads(content).items()]

    except json.decoder.JSONDecodeError:
        logging.error(
            f"vocab.json file {args.vocab} is not a json file."
        )
        raise ValueError(
            f'vocab.json file {args.vocab} is not a json file.'
        )

    if len(all_log_probs) == 0:
        raise ValueError(f'No valid audio files found at {args.data}')
    start_time = time.time()
    if args.no_parallel:
        for i in range(len(all_log_probs)):
            get_segments(
                all_log_probs[i],
                all_wav_paths[i],
                all_transcript_file[i],
                all_segment_file[i],
                vocabulary,
                args.window_len,
            )
    else:
        queue = multiprocessing.Queue(-1)

        listener = multiprocessing.Process(target=listener_process, args=(queue, listener_configurer, log_file, level))
        listener.start()
        workers = []
        for i in range(len(all_log_probs)):
            worker = multiprocessing.Process(
                target=worker_process,
                args=(
                    queue,
                    worker_configurer,
                    level,
                    all_log_probs[i],
                    all_wav_paths[i],
                    all_transcript_file[i],
                    all_segment_file[i],
                    vocabulary,
                    args.window_len,
                ),
            )
            workers.append(worker)
            worker.start()
        for w in workers:
            w.join()
        queue.put_nowait(None)
        listener.join()

    total_time = time.time() - start_time
    logger.info(f'Total execution time: ~{round(total_time/60)}min')
    logger.info(f'Saving logs to {log_file}')

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()