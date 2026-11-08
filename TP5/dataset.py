import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Transformations des images (ResNet attend des images 224x224 normalisées)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalisation standard ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Chargement du dataset Flickr30k
dataset = datasets.Flickr30k(
    root='/mnt/ssd/flickr30k/flickr30k-images',
    ann_file='/mnt/ssd/flickr30k/results_20130124.token',
    transform=transform
)

# 3. Séparation Train (80%) / Test (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

# 4. DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
