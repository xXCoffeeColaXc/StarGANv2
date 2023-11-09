from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
    
# NOTE Add seed for good evaluation images ?
class ACDCDataset(data.Dataset):
    def __init__(self, root_dir, selected_conditions=['daytime', 'fog'], transform=None, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        
        # Define the weather conditions and corresponding integer labels
        self.selected_conditions = selected_conditions
        self.condition_labels = {condition: i for i, condition in enumerate(selected_conditions)}
        print(self.condition_labels)
       
        self.transform = transform

        self.preprocess()

    def preprocess(self):
        # Collect all the image paths and corresponding integer labels
        self.img_paths = []
        self.labels = []
        for condition in self.selected_conditions:
            condition_path = os.path.join(self.root_dir, condition, self.mode)
            for folder in os.listdir(condition_path):
                if not folder.endswith('_ref'):  # Exclude the '_ref' folders
                    folder_path = os.path.join(condition_path, folder)
                    for img_file in os.listdir(folder_path):
                        if img_file.endswith('.jpg') or img_file.endswith('.png'):  # Assuming images are .jpg or .png
                            self.img_paths.append(os.path.join(folder_path, img_file))
                            self.labels.append(self.condition_labels[condition])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image) if self.transform else image
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)
    

def get_loader(image_dir, selected_attrs, image_size=128, batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""

    # Create Datalaoders
    train_transform = transforms.Compose([
            transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR),  # Resize the smallest side to 128 and maintain aspect ratio
            transforms.RandomCrop(image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # [0, 1] -> [-1, 1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
        ])
    # TODO val_loader

    dataset = ACDCDataset(root_dir=image_dir, selected_conditions=selected_attrs, transform=train_transform, mode=mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader