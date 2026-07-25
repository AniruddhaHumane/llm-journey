import torch.nn as nn
import torch

torch.manual_seed(42)

layer = nn.Linear(64, 64, dtype=torch.float32)
X = torch.randn((1, 64), dtype=torch.float32)
y_fp32 = layer(X)

# Assymmetric quantization
int8_scale = (2**8) - 1
scale = (layer.weight.max() - layer.weight.min()) / int8_scale
zero_point = torch.clip((-layer.weight.min() / scale).round(), 0, int8_scale)
int8_w = torch.clip((torch.round(layer.weight / scale) + zero_point), 0, int8_scale).to(torch.uint8)

# dequantizing it for inference
dequant_w = ((int8_w - zero_point) * scale).to(torch.float32)
y_fp32_dquant = X @ dequant_w.T + layer.bias

print((abs(y_fp32 - y_fp32_dquant)).mean(), abs(y_fp32 - y_fp32_dquant).max())
print(
    layer.weight.element_size() / int8_w.element_size(),
    layer.weight.nelement() / int8_w.nelement(),
)

# Symmetric quantization
scale = layer.weight.abs().max() / 127
zero_point = 0  # doesn't change
signed_int8_w = torch.clip((torch.round(layer.weight / scale)), -128, 127).to(torch.int8)
signed_dequant_w = (signed_int8_w * scale).to(torch.float32)
y_fp32_dquant = X @ signed_dequant_w.T + layer.bias

print((abs(y_fp32 - y_fp32_dquant)).mean(), abs(y_fp32 - y_fp32_dquant).max())
print(
    f"total memory consumption for fp32: {layer.weight.nelement() * layer.weight.element_size()} bytes"
)
print(f"total memory consumption for int8: {int8_w.nelement() * int8_w.element_size()} bytes")
print(
    f"total savings: {layer.weight.nelement() * layer.weight.element_size() / (int8_w.nelement() * int8_w.element_size())}x"
)

# # quantized inference
# y_int8 = (
#     X @ int8_w
# )  # unable to run this: expected m1 and m2 to have the same dtype, but got: float != signed char
