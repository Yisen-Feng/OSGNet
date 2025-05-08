import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from .data_utils import trivial_batch_collator, worker_init_reset_seed

datasets = {}

def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls

    return decorator


def make_dataset(name, is_training, split, val_jsonl_file, **kwargs):
    """
       A simple dataset builder
   """
    dataset = datasets[name](is_training, split, val_jsonl_file, **kwargs)
    return dataset


def make_data_loader(dataset, is_training, generator, batch_size, num_workers):
    """
        A simple dataloder builder
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,#(worker_init_reset_seed if is_training else None),
        # sampler=DistributedSampler(dataset),
        sampler=DistributedSampler(dataset,seed=torch.initial_seed()) if torch.distributed.is_initialized() else RandomSampler(dataset,generator=generator),
        drop_last=is_training,
        generator=generator,
        persistent_workers=num_workers > 0
    )
    return loader

