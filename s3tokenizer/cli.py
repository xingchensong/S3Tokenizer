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


import argparse


def get_args():
    parser = argparse.ArgumentParser(description='extract speech code')
    parser.add_argument('--wav_scp', required=True, type=str, help='each line contains `wav_name wav_path`')
    parser.add_argument('--device', required=True, type=str, help='cuda:x or cpu')
    parser.add_argument('--output', required=True, type=str, help='each line contains `wav_name wav_code`')
    parser.add_argument('--batch_size', required=True, type=int, help='batch size for inference')


def main():
    raise NotImplementedError("Commandline Usage is not supported now.")


if __name__ == "__main__":
    main()