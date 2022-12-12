import torch
import numpy as np
import random

def seed_worker(worker_id:int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

dataloader_confg = {
    'CIFAR': {
        'train': {
            'batch_size': 250,
            'shuffle': True,
            'drop_last': False,
        },
        'test': {
            'batch_size': 250,
            'shuffle': False,
            'drop_last': False,
        },
        'valid': {
            'batch_size': 250,
            'shuffle': False,
            'drop_last': False,
        },
        'memory': {
            'batch_size': 100,
            'shuffle': True,
            'drop_last': True,
        }
    },
    'CMNIST': {
        'train': {
            'batch_size': 250,
            'shuffle': True,
            'drop_last': False,
        },
        'test': {
            'batch_size': 250,
            'shuffle': False,
            'drop_last': False,
        },
        'valid': {
            'batch_size': 250,
            'shuffle': False,
            'drop_last': False,
        },
        'memory': {
            'batch_size': 100,
            'shuffle': True,
            'drop_last': True,
        }
    },
    'CelebA':{
        'train': {
            'batch_size': 125,
            'shuffle': True,
            'drop_last': False,
            'num_workers': 4,
            'prefetch_factor': 2,
            'worker_init_fn':seed_worker
        },
        'test': {
            'batch_size': 125,
            'num_workers': 4,
            'prefetch_factor': 2,
            'shuffle': False,
            'drop_last': False,
            'worker_init_fn':seed_worker
        },
        'valid': {
            'batch_size': 125,
            'num_workers': 4,
            'prefetch_factor': 2,
            'shuffle': False,
            'drop_last': False,
            'worker_init_fn':seed_worker
        },
        'memory': {
            'batch_size': 100,
            'shuffle': True,
            'drop_last': True,
            'num_workers': 4,
            'prefetch_factor': 2,
            'worker_init_fn':seed_worker
        }
    }
}