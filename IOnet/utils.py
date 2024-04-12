import os
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_tensor_images(tensor, title="Tensor", cmap=None):
    """
    Visualizes a tensor as an image. shape = (3, H, W)
    """
    tensor_np = tensor.detach().cpu().numpy()  # Convert tensor to numpy array
    # Transpose dimensions for imshow
    plt.imshow(np.transpose(tensor_np, (1, 2, 0)), cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def load_img_to_tensor(img_dir):
    '''
    Load images from a directory and convert them to tensors.

    Args:
        img_dir (str): Path to the directory containing images.

    Returns:
        torch.Tensor: Tensor containing the loaded images.
    '''
    # Input validation
    assert os.path.exists(img_dir), f"Directory {img_dir} does not exist."
    assert os.path.isdir(img_dir), f"{img_dir} is not a directory."

    # Load all image paths in the directory
    img_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]

    images = []
    # Load images to tensor
    for img_path in img_paths:
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        img /= 255.0
        print("Loaded image shape:", img.shape)
        assert img.shape == (
            3, 224, 224), f"Image shape is {img.shape} instead of (3, 224, 224)"
        images.append(img)

    return torch.stack(images)
