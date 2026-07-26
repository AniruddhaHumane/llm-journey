import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import numpy as np

images = list(Path("./data/images/test").glob("*.jpg"))


def preprocess_img(img: Image.Image):
    img = img.convert("RGB")
    kernel = np.ones((16, 16, 3))

    raw_img = np.array(img)
    print(f"raw image shape: {raw_img.shape}")
    n_tokens_raw = (raw_img.shape[0] // kernel.shape[0]) * (
        raw_img.shape[1] // kernel.shape[1]
    )
    print(f"token size for original image: {n_tokens_raw}")

    resized_img = np.array(img.resize((224, 224)))
    print(f"resized_img shape: {resized_img.shape}")

    # stride = 1 -> we want all blocks
    num_patches_x = resized_img.shape[0] // kernel.shape[0]
    num_patches_y = resized_img.shape[1] // kernel.shape[1]
    n_tokens_compressed = num_patches_x * num_patches_y

    print(f"total token count = {n_tokens_compressed}")

    patches = resized_img.reshape(
        num_patches_x, kernel.shape[0], num_patches_y, kernel.shape[1], kernel.shape[2]
    ).transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(
        num_patches_x * num_patches_y,
        kernel.shape[0] * kernel.shape[1] * kernel.shape[2],
    )
    print(f"patches shape np: {patches.shape}")

    patch_embed = nn.Conv2d(
        in_channels=3,
        out_channels=kernel.shape[0] * kernel.shape[1] * kernel.shape[2],
        kernel_size=16,
        stride=16,
    )
    tensor_img = (
        torch.tensor(resized_img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    )
    patches_tensor = patch_embed(tensor_img).squeeze(0).flatten(1).transpose(1, 0)

    print(f"patches shape torch: {patches_tensor.shape}")
    print(
        f"final_per_token_embdding_shape = {kernel.shape[0] * kernel.shape[1] * kernel.shape[2]}"
    )

    print(f"tokens comparison: {n_tokens_raw / n_tokens_compressed}")


preprocess_img(Image.open(images[0]))
