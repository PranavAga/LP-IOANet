# %%
"""
# Generating Low Resolution Images
"""

# %%
"""
## Imports
"""

# %%
from PIL import Image
import os
import torch
from torch import nn
import numpy as np
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

# %%
"""
## Loading images
"""

# %%
# TYPE = 'input'
# SET = 'test'

# %%
# %time
def load_images_from_folder(folder_path):
    images = []
    names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            images.append(img)
            names.append(filename)
    return [images,names]

folder_path = f"./"
loaded_images = load_images_from_folder(folder_path)

# %%
print(len(loaded_images[0]), flush=True)
print(loaded_images[1][0], flush=True)
print(loaded_images[0][0].size, flush=True)

# %%
transform = transforms.Compose([
    transforms.Resize((1024,768)),
    transforms.ToTensor(),           # Convert images to tensors
])

# %%
# Convert PIL images to tensors
# %time
tensor_images = [transform(img) for img in loaded_images[0]]
tensor_images = torch.stack(tensor_images)

# %%
print(tensor_images[0].shape,flush=True)

# %%
"""
## Generating down-sampled images
"""

# %%
L=2
I=[]
I.append(tensor_images)

for l in range(L):
    Il = nn.functional.interpolate(tensor_images,size=[I[-1].shape[2]//2,I[-1].shape[3]//2],mode='nearest-exact')
    I.append(Il)

# %%
print(I[-1].shape,flush=True)
# %%
"""
## Saving the generated images
"""

# %%
def save_image(image_data,output_folder,name):

    image_data_scaled = (image_data * 255).astype(np.uint8)
    image = Image.fromarray(image_data_scaled)
    image.save(os.path.join(output_folder, f'{name}'))
# %%
for i,img_data_tensor in enumerate(I[-1]):
    low_res_folder = f"./lowres"
    if not os.path.exists(low_res_folder):
        os.makedirs(low_res_folder)
    prevname = loaded_images[1][i]
    if i==0:
        print(f'Final shape: {img_data_tensor.shape}')
    save_image(np.swapaxes(np.swapaxes(img_data_tensor,0, 2),0,1).numpy(),low_res_folder,f'{prevname[:len(prevname)-4]}-{img_data_tensor.shape[2]},{img_data_tensor.shape[1]}{prevname[len(prevname)-4:]}')
# %%
