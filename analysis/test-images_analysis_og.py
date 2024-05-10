# %%
"""
 ## Imports
"""

# %%
print("Importing lib....", flush=True)

from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import torch
from tqdm import tqdm

# %%
""" Normalization class """

norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]
# unromalize the image
def unormalize_np_image(image, mean=norm_mean, std=norm_std):
    return np.clip((image * std + mean) * 255, 0, 255).astype(np.uint8)


# %%
""" ## Load Data """
print("Loading Data", flush=True)

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            images.append(Image.open(image_path))
    return images

folder_path_shadow = f"./SD7K/train/input"
folder_path_removed = f"./SD7K/train/target"

n_images =4000
img_shadow_raw = load_images_from_folder(folder_path_shadow)[:n_images]
print('loaded images')
img_removed_raw = load_images_from_folder(folder_path_removed)[:n_images]
print('loaded images')

transform = transforms.Compose([
    transforms.Resize((1024,768)),  # Ensure the size
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=norm_mean, std=norm_std)
])

img_shadow_ = [transform(img) for img in img_shadow_raw]
del img_shadow_raw
img_shadow = torch.stack(img_shadow_)
del img_shadow_
print('nomralized shadow images')

img_removed_ = [transform(img) for img in img_removed_raw]
del img_removed_raw
img_removed = torch.stack(img_removed_)
del img_removed_
print('nomralized removed images')


# %%
# dump the data
torch.save(img_shadow, "./img_shadow.pth")
torch.save(img_removed, "./img_removed.pth")

# %%
print("\n>\t","loading temp files", flush=True)
img_shadow = torch.load("./img_shadow.pth")
img_removed = torch.load("./img_removed.pth")

# %%
"""
## 

"""

# %%
batch_size = 32
test_dataset = torch.utils.data.TensorDataset(img_shadow, img_removed)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# %%
total_acc = 0
for x_test, y_test in tqdm(test_loader):
    metric = SSIM()
    acc = metric(x_test,y_test)
    total_acc+= acc*x_test.shape[0]

print(img_shadow.shape[0])
print(total_acc.item()/img_shadow.shape[0])

# %%
