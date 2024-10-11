# Copyright (c) 2024 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Example Usage
cpu:

s3tokenizer --wav_scp xxx.scp \
            --device "cpu" \
            --output_dir "./" \
            --batch_size 32

gpu:

torchrun --nproc_per_node=8 --nnodes=1 \
     --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    `which s3tokenizer` --wav_scp xxx.scp \
                --device "cuda" \
                --output_dir "./" \
                --batch_size 32

"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

import s3tokenizer


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    print('Inference on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def init_dataset_and_dataloader(files, batch_size, num_workers, prefetch):

    def decode_wav(sample):
        key, filepath = sample['line'].split()
        audio = s3tokenizer.load_audio(filepath)
        return {
            'key': key,
            'wav': audio,
        }

    def compute_feature(sample):
        wav = sample['wav']
        mel = s3tokenizer.log_mel_spectrogram(wav)
        sample['mel'] = mel
        return sample

    def filter_by_length(sample, max_seconds=30):
        wav = sample['wav']
        if wav.shape[0] / 16000 <= 30:
            return True
        return False

    def padding(data):
        keys = [sample['key'] for sample in data]
        mels_list = [sample['mel'] for sample in data]
        mels, mels_lens = s3tokenizer.padding(mels_list)
        return {
            'keys': keys,
            'mels': mels,
            'mels_lens': mels_lens,
        }

    from s3tokenizer.input_pipeline.datapipes import WenetRawDatasetSource
    dataset = WenetRawDatasetSource(files, cycle=1, shuffle=False)

    dataset = dataset.map(decode_wav)
    dataset = dataset.filter(filter_by_length)
    dataset = dataset.map(compute_feature)
    dataset = dataset.batch(batch_size, wrapper_class=padding)

    dataloader = DataLoader(dataset,
                            batch_size=None,
                            num_workers=num_workers,
                            persistent_workers=True,
                            prefetch_factor=prefetch)
    return dataset, dataloader


def get_args():
    parser = argparse.ArgumentParser(description='extract speech code')
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        choices=["speech_tokenizer_v1", "speech_tokenizer_v1_25hz"],
        help='model version')
    parser.add_argument('--wav_scp',
                        required=True,
                        type=str,
                        help='each line contains `wav_name wav_path`')
    parser.add_argument('--device',
                        required=True,
                        type=str,
                        choices=["cuda", "cpu"],
                        help='device for inference')
    parser.add_argument('--output_dir',
                        required=True,
                        type=str,
                        help='dir to save result')
    parser.add_argument('--batch_size',
                        required=True,
                        type=int,
                        help='batch size (per-device) for inference')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='workers for dataloader')
    parser.add_argument('--prefetch',
                        type=int,
                        default=5,
                        help='prefetch for dataloader')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda":
        assert (torch.cuda.is_available())
        world_size, _, rank = init_distributed()
    else:
        world_size, _, rank = 1, 0, 0

    device = torch.device(args.device)
    model = s3tokenizer.load_model(args.model).to(device)
    _, dataloader = init_dataset_and_dataloader(args.wav_scp, args.batch_size,
                                                args.num_workers,
                                                args.prefetch)

    if args.device == "cuda":
        model = model.cuda()

    if rank == 0:
        progress_bar = tqdm(desc="Processing", dynamic_ncols=True, unit="wavs")

    writer = open(f"{args.output_dir}/part_{rank + 1}_of_{world_size}", "w")
    for batch in dataloader:
        keys, mels, mels_lens = batch['keys'], batch['mels'], batch[
            'mels_lens']
        codes, codes_lens = model(mels.to(device), mels_lens.to(device))
        for i, k in enumerate(keys):
            code = codes[i, :codes_lens[i].item()].tolist()
            writer.write(
                json.dumps({
                    "key": k,
                    "code": code
                }, ensure_ascii=False) + "\n")
        if rank == 0:
            progress_bar.update(world_size * len(keys))

    if rank == 0:
        progress_bar.close()
    writer.close()
    if args.device == "cuda":
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
