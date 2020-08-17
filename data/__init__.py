import importlib
import torch
import torch.multiprocessing as mp
import horovod.torch as hvd


def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls_ in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls_
            
    if dataset is None:
        raise ValueError("In %s.py, there should match %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def create_dataloader(cfg):
    dataset = find_dataset_using_name(cfg.dataset_name)
    instance = dataset(cfg)
    print("Dataset [%s] of size %d was created." % (type(instance).__name__, len(instance)))

    if cfg.phase == 'train':
        allreduce_batch_size = cfg.batch_size * cfg.batches_per_allreduce
        torch.set_num_threads(cfg.num_workers)
        kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True, 'drop_last': True}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'
        
        # split dataset
        val_size = int(len(instance) * 0.2) if cfg.validation else 0
        train_size = len(instance) - val_size
        lengths = [train_size, val_size]
        train_dataset, val_dataset = torch.utils.data.random_split(instance, lengths)
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=allreduce_batch_size, sampler=train_sampler, **kwargs)

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, sampler=val_sampler, **kwargs)
        
        print("dataset [%s] was splited randomly (%d, %d)" % (type(instance).__name__, len(train_dataset), len(val_dataset)))
        return train_sampler, train_loader, val_sampler, val_loader
    elif cfg.phase == 'val' or cfg.phase == 'test':
        test_loader = torch.utils.data.DataLoader(instance, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        return test_loader
    else:
        raise ValueError('In cfg.phase, we expect train or val.')
