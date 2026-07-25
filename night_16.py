import torch

a = torch.tensor([0.0000000001, 500000001], dtype=torch.float32)
print(a, a.element_size())

a16 = a.to(dtype=torch.float16)
print(a16, a16.element_size(), a - a16)

ab16 = a.to(dtype=torch.bfloat16)
print(ab16, ab16.element_size(), a - ab16)

'''
BF16 maintained the range of numbers and unlike fp16 the large value did not turn into inf
However the numbers diverged significantly. There is a prcision error of 10^5 when converting the big number back 
Hence BF16 has the range but precision bits are significantly lower resulting in higher error when converting back

'''

scale = a.abs().max() / 127 # range of int_8
int_8 = torch.clamp(torch.round(a / scale), -127, 127).to(torch.int8)
print(int_8, int_8.element_size())
dequant_8 = int_8.float() * scale
print(dequant_8, dequant_8.element_size(), a - dequant_8)

scale = a.abs().max() / 7 # range of int_4
int_4 = torch.clamp(torch.round(a / scale), -7, 7).to(torch.int8) # there is no int4
print(int_4, int_4.element_size()) # still returns 1 byte
dequant_4 = int_4.float() * scale
print(dequant_4, dequant_4.element_size(), a - dequant_4)