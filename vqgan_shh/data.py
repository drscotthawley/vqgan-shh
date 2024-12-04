import os
import math
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random
from PIL import Image


def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list  # list of allowed file extensions
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in ext:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


class PairDataset(Dataset):
    "li'l thing that grabs two images at a time"
    def __init__(self, base_dataset):
        self.dataset, self.indices = base_dataset, list(range(len(base_dataset)))
        
    def __len__(self): 
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # Get source image and class
        source_img, source_class = self.dataset[idx]
        target_idx = idx # random.choice(self.indices) # for now just do reconstruction.
        target_img, target_class = self.dataset[target_idx]
        
        return source_img, source_class, target_img, target_class


def create_loaders(batch_size=32, image_size=128, shuffle_val=True, data_path=None):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        #transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if data_path is None:
        # Use Oxford Flowers dataset
        train_base = datasets.Flowers102(root='./data', split='train', transform=train_transforms, download=True)
        val_base = datasets.Flowers102(root='./data', split='val', transform=val_transforms, download=True)
        train_dataset = PairDataset(train_base)
        val_dataset = PairDataset(val_base)
    else:
        # Custom directory handling
        _, all_files = fast_scandir(data_path, ['jpg', 'jpeg', 'png'])
        random.shuffle(all_files)  # Randomize order
        
        # Split into train/val (90/10 split)
        split_idx = int(len(all_files) * 0.9)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        class ImageFolderDataset(Dataset):
            def __init__(self, file_list, transform=None):
                self.files = file_list
                self.transform = transform
                
            def __len__(self):
                return len(self.files)
                
            def __getitem__(self, idx):
                img = Image.open(self.files[idx]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, 0  # Return 0 as class label since we don't have classes
        
        train_dataset = PairDataset(ImageFolderDataset(train_files, train_transforms))
        val_dataset = PairDataset(ImageFolderDataset(val_files, val_transforms))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=8)
    
    return train_loader, val_loader
