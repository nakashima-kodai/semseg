import os
import random
import numpy
import math
import torch
import hydra
import horovod.torch as hvd
from apex import amp
from omegaconf import DictConfig
from data import create_dataloader
from models import create_model
from utils.visualizer import Visualizer
from utils.iteration_counter import IterationCounter
from utils.metrics import AverageMeter


def update_lr(cfg, epoch, optimizer):
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % cfg.decay_epoch_freq == 0:
        new_lr = current_lr * 0.1
    else:
        new_lr = current_lr
    
    if new_lr != current_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr * hvd.size()


def inverse_normalization(image):
    x = image.new(*image.size())
    x[:, 0, :, :] = image[:, 0, :, :] * 0.229 + 0.485
    x[:, 1, :, :] = image[:, 1, :, :] * 0.224 + 0.456
    x[:, 2, :, :] = image[:, 2, :, :] * 0.225 + 0.406
    return x


@hydra.main(config_path='./configs/config.yaml')
def main(cfg : DictConfig):
    # initialize ditributed mode
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # fix random seed
    random.seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # set cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # set visualizer
    visualizer = Visualizer(cfg) if hvd.rank() == 0 else None

    # set dataloader
    train_sampler, train_loader, val_sampler, val_loader = create_dataloader(cfg)

    # set model
    model = create_model(cfg)
    if visualizer is not None:
        visualizer.save_architecture(model)
    model.cuda()

    # set optimizer
    lr_scaler = cfg.batches_per_allreduce * hvd.size() if not cfg.use_adasum else 1
    if cfg.use_adasum and hvd.nccl_built():
        lr_scaler = cfg.batches_per_allreduce * hvd.local_size()
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=cfg.lr * lr_scaler,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )

    compression = hvd.Compression.fp16 if cfg.fp16_allreduce else hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=cfg.batches_per_allreduce,
        op=hvd.Adasum if cfg.use_adasum else hvd.Average
    )

    # broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Apex
    if cfg.use_fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # set criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.num_classes).cuda()


    iter_counter = IterationCounter(cfg, len(train_loader.dataset))
    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = AverageMeter('train_loss')
        for idx, data in enumerate(train_loader):
            iter_counter.record_one_iteration()
            
            image = data['image'].cuda()
            label = data['label'].cuda().squeeze(dim=1)

            optimizer.zero_grad()
            for i in range(0, len(data), cfg.batch_size):
                image_batch = image[i:i+cfg.batch_size]
                label_batch = label[i:i+cfg.batch_size]
                score = model(image_batch)
                loss = criterion(score, label_batch.long())
                train_loss.update(loss)
                loss.div_(math.ceil(float(len(data)) / cfg.batch_size))

                if cfg.use_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        optimizer.synchronize()
                    with optimizer.skip_synchronize():
                        optimizer.step()
                else:
                    loss.backward()
                    optimizer.step()

            if iter_counter.needs_logging_losses() and visualizer is not None:
                gs = iter_counter.global_steps
                lr = optimizer.param_groups[0]['lr']
                log_items = {'epoch': epoch, 'lr': lr, 'train_loss': train_loss.avg}
                visualizer.log_losses(log_items, gs)

            if iter_counter.needs_logging_visuals() and visualizer is not None:
                gs = iter_counter.global_steps
                img = inverse_normalization(data['image'])
                pred_label = torch.max(score, dim=1, keepdim=True)[1]
                visuals = {'image': img, 'label': data['label'], 'pred_label': pred_label}
                visualizer.log_visuals(visuals, gs)
            

        update_lr(cfg, epoch, optimizer)

        if cfg.validation:
            model.eval()
            val_loss = AverageMeter('val_loss')
            for data in val_loader:
                image = data['image'].cuda()
                label = data['label'].cuda().squeeze(dim=1)

                with torch.no_grad():
                    score = model(image)
                loss = criterion(score, label.long())
                val_loss.update(loss)
            
            if visualizer is not None:
                log_items = {'val_loss': val_loss.avg}
                visualizer.logging(log_items, iter_counter.global_steps)

        if iter_counter.needs_saving() and hvd.rank() == 0:
            save_name = '{}_bs{:03d}_lr{:.2e}_ep{:03d}.pth'.format(cfg.model_name, cfg.batch_size, cfg.lr, epoch)
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if cfg.use_fp16:
                state['amp'] = amp.state_dict()
            torch.save(state, os.path.join(os.getcwd(), save_name))

        iter_counter.record_epoch_end()

if __name__ == '__main__':
    main()
