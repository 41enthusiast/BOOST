import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, index
        

# Define the path to your dataset
data_dir = 'data/PACS'
BSZ = 4

# Define the transformation to convert images to tensors
transform = transforms.ToTensor()

# Load the dataset using ImageFolder or a custom Dataset class
dataset = CustomImageFolder(root=data_dir, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=BSZ, shuffle=False, num_workers=4)

# Initialize variables to store the sum and sum of squares
mean = torch.zeros(3)
std = torch.zeros(3)
n_images = 0

# Iterate over the dataset
for images, _, __ in tqdm(dataloader):
    # Accumulate the sum and squared sum of pixel values across the dataset
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_images += batch_samples

# Final mean and standard deviation calculation
mean /= n_images
std /= n_images

print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")

mean_channels = mean.tolist()
std_channels = std.tolist()
kaokore_transform1 = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(mean_channels), tuple(std_channels)),
    ])


dataset = CustomImageFolder(root=data_dir, transform=kaokore_transform1)
print(len(dataset), len(dataset[0]), dataset[0][0].shape, dataset[0][1])

# Define split sizes
train_size = 0.7  # 70% for training
val_size = 0.15   # 15% for validation
test_size = 0.15  # 15% for testing
total_size = len(dataset)
train_len = int(train_size * total_size)
val_len = int(val_size * total_size)
test_len = total_size - train_len - val_len  # Ensures the sizes add up

norm_std = std

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
test_loader_out = DataLoader(test_dataset, batch_size = BSZ)
train_loader_out = DataLoader(train_dataset, batch_size = BSZ)

if len(data_dir.split('/')) != 3:
    class_names = {0: 'art_painting', 1: 'photo', 2: 'cartoon', 3: 'sketch'}
else:
    class_names = {0: 'dog', 1: 'elephant', 2: 'giraffe', 3: 'guitar', 4: 'horse', 5: 'house', 6: 'person'}

print(len(train_dataset), len(val_dataset), len(test_dataset))
print('Label set', set([i[1] for i in train_dataset]), set([i[1] for i in test_dataset]))
# print(train_dataset[0][0], train_dataset[0][1])
print(train_dataset[0][0].shape, train_dataset[0][0].dtype)
print(train_dataset[0][0].max(), train_dataset[0][0].min(), train_dataset[0][0].mean(), train_dataset[0][0].std())
