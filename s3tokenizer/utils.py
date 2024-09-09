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
        if "quantizer" in k:  # vq
            if k == "/quantizer/rq/model/layers.0/_codebook/Pow_1":
                new_weight_dict["quantizer._codebook.embed"] = weights_dict[k] 
        elif "positional_embedding" in k:  # positional emb
            new_weight_dict[k] = weights_dict[k]
        elif "conv" in k:  # 1/2 subsample
            new_weight_dict[k] = weights_dict[k]
        else:  # transformer blocks
            assert "blocks" in k
            new_k = (
                k[1:]
                .replace('/', '.')
                .replace('MatMul', 'weight')
                .replace('Add_1', 'bias')
                .replace('Mul', 'weight')
                .replace('Add', 'bias')
                .replace('mlp.mlp', 'mlp')
            )
            new_weight_dict[f"encoder.{new_k}"] = weights_dict[k] 
    return new_weight_dict


def onnx2torch(onnx_path: str, torch_path: str = None, verbose: bool = False):
    onnx_model = onnx.load(onnx_path)
    weights_dict = {}
    initializer_map = {initializer.name: initializer for initializer in onnx_model.graph.initializer}
    for node in onnx_model.graph.node:
        for input_name in node.input:
            if input_name in initializer_map:
                initializer = initializer_map[input_name]
                if input_name == "onnx::Conv_1519":
                    weight_name = "encoder.conv1.weight"
                elif input_name == "onnx::Conv_1520":
                    weight_name = "encoder.conv1.bias"
                elif input_name == "onnx::Conv_1521":
                    weight_name = "encoder.conv2.weight"
                elif input_name == "onnx::Conv_1522":
                    weight_name = "encoder.conv2.bias"
                elif input_name == "encoders.positional_embedding":
                    weight_name = "encoder.positional_embedding"
                else:
                    weight_name = node.name
                weight_array = onnx.numpy_helper.to_array(initializer).copy()
                weight_array.flags.writeable = True
                weight_tensor = torch.from_numpy(weight_array)
                if len(weight_tensor.shape) > 2 or weight_name == "encoder.positional_embedding":
                    weights_dict[weight_name] = weight_tensor
                else:
                    weights_dict[weight_name] = weight_tensor.t()
    new_weights_dict = rename_weights(weights_dict)
    if verbose:
        for k, v in new_weights_dict.items():
            print(f"{k} : {v.shape} {v.dtype}")
        print(f"PyTorch weights saved to {torch_path}")
    if torch_path:
        torch.save(new_weights_dict, torch_path)
    else:
        return new_weights_dict