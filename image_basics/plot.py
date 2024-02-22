"""PLotting routines
"""
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as tvF


def plot_images(images, num_cols=5):
    # coerce to 2d list
    if not isinstance(images, list):
        images = [images]
    if not isinstance(images[0], list):
        images = [images[i : i + num_cols] for i in range(0, len(images), num_cols)]
    N, M = len(images), max([len(row) for row in images])

    fig, axs = plt.subplots(N, M, squeeze=False)
    for i in range(N):
        for j, img in enumerate(images[i]):
            img_pil = tvF.to_pil_image(tvF.to_dtype(img, dtype=torch.uint8, scale=True))
            ax = axs[i, j]
            ax.imshow(img_pil)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
