import torch
from torch import nn

from s3tokenizer.model_v2 import FSQCodebook


class FakeProjectDown(nn.Module):

    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps = x.shape[:2]
        assert self.output.shape[:2] == (batch, steps)
        return self.output.to(x.device)


def test_fsq_code_packing_returns_exact_token_ids():
    codebook = FSQCodebook(dim=8, level=3)
    digits = torch.tensor(
        [[[2, 0, 0, 0, 0, 2, 0, 0], [1, 0, 0, 0, 0, 2, 0, 0]]],
        dtype=torch.int64,
    )
    pre_tanh = torch.where(
        digits == 2,
        torch.full_like(digits, 20),
        torch.where(digits == 1, torch.zeros_like(digits), torch.full_like(digits, -20)),
    ).to(torch.float32)
    codebook.project_down = FakeProjectDown(pre_tanh)

    dummy_hidden = torch.zeros(1, 2, 8)
    packed = codebook.encode(dummy_hidden)

    assert torch.equal(packed, torch.tensor([[488, 487]], dtype=torch.int32))


def test_cuda_float_pow_can_be_non_integer():
    if not torch.cuda.is_available():
        return

    powers = torch.pow(3, torch.arange(8, device="cuda", dtype=torch.float32)).cpu()

    # On some CUDA stacks, 3**5 becomes 242.99998 instead of 243.0.
    # This test documents the motivation for avoiding float pow in code packing.
    assert powers[5].item() <= 243.0
