# -*- coding: utf-8 -*-
# @Time  : 2022/5/6 23:24
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : manifest_for_align.py
import os


def generate_manifest_from_wav2vec(tsv_path, label_path, output, split=0.8):
    with open(tsv_path, 'r', encoding='utf-8') as tsv:
        with open(label_path, 'r', encoding='utf-8') as label:
            tsv.readline()
            path_nframes = tsv.readlines()
            labels = label.readlines()
    with open(output, 'r', encoding='utf-8') as output:
        output.write('\n'.join(path_nframes[:int(len(path_nframes) * split)]))
        output.write('\n')
        output.write('\n'.join(labels[:int(len(labels) * split)]))

    datasets = zip(path_nframes, labels)
    index = int(len(datasets) * split)
    train_datasets = datasets[:index]
    dev_datasets = datasets[index:]
    write_file(os.path.join(output, 'train.tsv'), train_datasets)
    write_file(os.path.join(output, 'dev.tsv'), dev_datasets)


def write_file(path, datasets, max_nframes=200000):
    result = []
    with open(path, 'w', encoding='utf-8') as path:
        for path_nframes, labels in datasets:
            file_path, nframes = path_nframes.strip().split('\t')
            if max_nframes > int(nframes):
                result.append(f"{file_path}\t{labels.strip()}")
        path.write('\n'.join(result))


def generate_manifest(tsv_path, train_path, output, max_nframes=200000):
    with open(tsv_path, 'r', encoding='utf-8') as tsv:
        tsv.readline()
        path_nframes = tsv.readlines()
    total = {}
    with open(train_path, 'r', encoding='utf-8') as train:
        train.readline()
        path_label = train.readlines()
        for i in path_label:
            if i == '\n':
                continue
            path, label = i.strip().split('\t')
            total[path] = label
    result = []
    for item in path_nframes:
        file_path, nframes = item.strip().split('\t')
        if max_nframes > int(nframes):
            if file_path in total.keys():
                result.append((item, total[file_path]))
    write_file(os.path.join(output, 'train_less.tsv'), result, 100000)




if __name__ == '__main__':
    generate_manifest('/home/data/dataset/vietnamese/wav2vec2_manifest/train.tsv',
                      '/home/data/SpeechTextDatasetConstruct/mainfest_for_attention_alignment/train.tsv',
                      '/home/data/SpeechTextDatasetConstruct/mainfest_for_attention_alignment')
