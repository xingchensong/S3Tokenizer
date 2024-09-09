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


import torch
import onnx


def rename_weights(weights_dict: dict):
    new_weight_dict = {}
    for k in weights_dict.keys():
        new_k = k.replace('/', '.')
        if "blocks" in k:
            new_weight_dict[f"encoder.{new_k}"] = weights_dict[k] 
        elif "quantizer" in k:
            new_weight_dict[f"{new_k}"] = weights_dict[k] 
    return new_weight_dict


def onnx2torch(onnx_path: str, torch_path: str):
    onnx_model = onnx.load(onnx_path)
    weights_dict = {}
    initializer_map = {initializer.name: initializer for initializer in onnx_model.graph.initializer}
    for node in onnx_model.graph.node:
        for input_name in node.input:
            if input_name in initializer_map:
                initializer = initializer_map[input_name]
                weight_name = node.name
                weight_tensor = torch.from_numpy(onnx.numpy_helper.to_array(initializer))
                weights_dict[weight_name] = weight_tensor
    new_weights_dict = rename_weights(weights_dict)
    for k, v in weights_dict.items():
        print(f"{k} : {v.shape} {v.dtype}")
    torch.save(new_weights_dict, torch_path)
    print(f"PyTorch weights saved to {torch_path}")