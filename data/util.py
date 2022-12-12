import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torchvision.datasets.celeba import CelebA
from data.attr_dataset import AttributeDataset, AttributeDataset_bffhq
import os
import torch
from glob import glob
from PIL import Image
from data.celeba import CelebA

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item

    
transforms = {
    "ColoredMNIST": {
        "train": T.Compose([T.ToTensor()]),
        "eval": T.Compose([T.ToTensor()])
        },
    "CorruptedCIFAR10": {
        "train_aug": T.Compose(
            [
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "train": T.Compose(
            [
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "eval": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
    "Shapes3D": {
        "train": T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.Resize((32, 32)),
            T.ToTensor(),
        ]),
        "eval": T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.Resize((32, 32)),
            T.ToTensor(),
        ]),
    },
    "CelebA": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    },
    "bffhq": {
        "train": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224,224)), T.ToTensor()])
        }
}


transforms_preprcs = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        },
    "cifar10c": {
        "train": T.Compose(
            [
                T.RandomCrop(32, padding=4),
                # T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
}

transforms_preprcs_ae = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    },
    "cifar10c": {
        "train": T.Compose(
            [
                # T.RandomCrop(32, padding=4),
                T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
}

class bFFHQDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.root = root

        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict
            self.attr = torch.LongTensor([[int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])] for index in range(len(self.data))])

        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
            self.attr = torch.LongTensor([[int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])] for index in range(len(self.data))])

        elif split=='test':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
                if target_label != bias_label:
                    data_conflict.append(path)
            self.data = data_conflict
            self.attr = torch.LongTensor([[int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])] for index in range(len(self.data))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = self.attr[index]
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr

def get_dataset_bffhq(dataset, data_dir, dataset_split, transform_split, percent, use_preprocess=None, image_path_list=None, use_type0=None, use_type1=None):
    dataset_category = dataset.split("-")[0]
    if use_preprocess:
        transform = transforms_preprcs[dataset_category][transform_split]
    else:
        transform = transforms[dataset_category][transform_split]
    dataset_split = "valid" if (dataset_split == "eval") else dataset_split

    if dataset == "bffhq":
        root = data_dir + f"/bffhq/{percent}"
        # print(root)
        dataset = bFFHQDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)
    else:
        print('wrong dataset ...')
        import sys
        sys.exit(0)

    return AttributeDataset_bffhq(dataset, None, transform)

def get_dataset(dataset_tag, data_dir, dataset_split, transform_split, add = False):
    dataset_category = dataset_tag.split("-")[0]
    data_dir = '../Bias_SSL/datasets'
    root = os.path.join(data_dir, dataset_tag)
    if dataset_tag != 'bffhq':
        transform = transforms[dataset_category][transform_split]
        dataset_split = "valid" if (dataset_split == "eval") else dataset_split
    if dataset_tag == "bffhq":
        if dataset_split == transform_split and dataset_split == 'eval':
            dataset_split = 'test'
            transform_split = 'valid'
        elif add == True and dataset_split == 'train':
            dataset_split = 'valid'
            transform_split = 'valid'
   
        dataset = get_dataset_bffhq('bffhq', data_dir, dataset_split, transform_split,percent='0.5pct',use_preprocess=True, use_type0=None, use_type1=None)
    elif dataset_tag == 'CelebA':
        celeba_root = '../Bias_SSL/datasets/CelebA'
        dataset = CelebA(celeba_root, split=dataset_split, transform=transform, target_type='attr', download=False)
    else:
        dataset = AttributeDataset(
            root=root, split=dataset_split, transform=transform
        )

    return dataset


