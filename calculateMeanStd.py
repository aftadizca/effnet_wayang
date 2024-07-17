import torch
import torchvision
import numpy as np

# Assuming you have a dataset class with an `__getitem__` method
# that returns a PIL image, and a list of file paths `files`
dataset = "WayangKulitDataset\\1. arjuna"  # Initialize your dataset

# Initialize arrays to store means and stds for each channel
means = [0, 0, 0]  # RGB channels
stds = [0, 0, 0]

for image in dataset:
    print(image)
    # Convert PIL image to tensor
    img_tensor = torchvision.transforms.ToTensor()(image)
    # Calculate mean and std for each channel
    for c in range(3):
        channel = img_tensor[:, :, c]
        means[c] += channel.mean()
        stds[c] += (channel - channel.mean()) ** 2

# Calculate final means and stds
means = [m / len(dataset) for m in means]
stds = [np.sqrt(s / len(dataset)) for s in stds]

print(means)
print(stds)
