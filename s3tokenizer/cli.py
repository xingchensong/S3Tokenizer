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

s3tokenizer --cuts_path xxx.jsonl.gz \
    --device "cpu" \
    --output_dir "." \
    --batch_size 32

gpu:

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --rdzv_id=2024 \
    --rdzv_backend="c10d" \
    --rdzv_endpoint="localhost:0" \
    `which s3tokenizer` \
        --cuts_path xxx.jsonl.gz \
        --device "cuda" \
        --output_dir "." \
        --batch_size 32

"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from lhotse import CutSet
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

import s3tokenizer


class AudioDataset(Dataset):
    def __init__(self, cuts_path):
        self.data = (
            CutSet.from_file(cuts_path).filter(lambda c: c.duration <= 30).to_eager()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cut = self.data[idx]
        key = cut.id
        audio = cut.resample(16000).load_audio().squeeze()
        mel = s3tokenizer.log_mel_spectrogram(audio)
        return key, mel


def collate_fn(batch):
    keys = [item[0] for item in batch]
    mels = [item[1] for item in batch]
    mels, mels_lens = s3tokenizer.padding(mels)
    return keys, mels, mels_lens


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    print('Inference on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def get_args():
    parser = argparse.ArgumentParser(description='extract speech code')
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        choices=[
                            "speech_tokenizer_v1", "speech_tokenizer_v1_25hz",
                            "speech_tokenizer_v2_25hz"
                        ],
                        help='model version')
    parser.add_argument("--cuts_path",
                        required=True,
                        type=str,
                        help='path to manifests')
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
        assert torch.cuda.is_available()
        world_size, local_rank, rank = init_distributed()
    else:
        world_size, local_rank, rank = 1, 0, 0

    device = torch.device(args.device)
    model = s3tokenizer.load_model(args.model).to(device)
    dataset = AudioDataset(args.cuts_path)

    if args.device == "cuda":
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])
        sampler = DistributedSampler(dataset,
                                     num_replicas=world_size,
                                     rank=rank)
    else:
        sampler = None

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch,
                            collate_fn=collate_fn)

    total_steps = len(dataset)

    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Processing", unit="wavs")

    writer = open(f"{args.output_dir}/part_{rank + 1}_of_{world_size}", "w")
    for keys, mels, mels_lens in dataloader:
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
