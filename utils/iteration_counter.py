import os
import time
import numpy as np
import horovod.torch as hvd


class IterationCounter():
    def __init__(self, cfg, dataset_size):
        self.cfg = cfg
        self.batch_size = cfg.batch_size * cfg.batches_per_allreduce * hvd.size()
        self.first_epoch = 1
        self.total_epochs = cfg.epochs
        self.epoch_iter = 0  # iter number within each epoch
        self.iter_record_path = os.path.join(os.getcwd(), 'iter.txt')
        if cfg.phase=='train' and cfg.continue_train:
            try:
                self.first_epoch, self.epoch_iter = np.loadtxt(self.iter_record_path, delimiter=',', dtype=int)
                print('Resuming from epoch %d at iteration %d' % (self.first_epoch, self.epoch_iter))
            except:
                print('Could not load iteration record at %s. Starting from beginning.' % self.iter_record_path)

        self.global_steps = (self.first_epoch - 1) * dataset_size + self.epoch_iter

    def training_epochs(self):
        return range(self.first_epoch, self.total_epochs + 1)

    def record_epoch_start(self, epoch):
        self.epoch_start_time = time.time()
        self.epoch_iter = 0
        self.current_epoch = epoch

    def record_one_iteration(self):
        self.global_steps += self.batch_size
        self.epoch_iter += self.batch_size

    def record_epoch_end(self):
        current_time = time.time()
        self.time_per_epoch = current_time - self.epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' % (self.current_epoch, self.total_epochs, self.time_per_epoch))

    def record_current_iter(self):
        np.savetxt(self.iter_record_path, (self.current_epoch, self.epoch_iter), delimiter=',', fmt='%d')
        print('Saved current iteration count at %s.' % self.iter_record_path)

    def needs_saving(self):
        return (self.current_epoch % self.cfg.save_epoch_freq) == 0

    def needs_logging_losses(self):
        return (self.global_steps % self.cfg.log_loss_freq) < self.batch_size

    def needs_logging_visuals(self):
        return (self.global_steps % self.cfg.log_visual_freq) < self.batch_size
