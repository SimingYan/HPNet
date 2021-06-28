import os
import numpy as np
import argparse
import torch
from torch import nn
import torch.utils.data
import matplotlib.pyplot as plt
from utils.tf_visualizer import Visualizer as TfVisualizer
from utils.main_utils import parameter_count, get_model_module
from collections import defaultdict
import time

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def cuda_time():
    torch.cuda.synchronize()
    return time.time()


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt

        self.build_workspace()
        self.build_dataloader()
        self.build_model_optimizer()

        # TFBoard visualizer
        self.TRAIN_VISUALIZER = TfVisualizer(self.LOG_DIR, 'train')
        self.TEST_VISUALIZER = TfVisualizer(self.LOG_DIR, 'test')

    def build_workspace(self):

        # Prepare LOG_DIR
        self.LOG_DIR = self.opt.log_dir
        MODEL_NAME = self.LOG_DIR.split('/')[-1]
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)

        if self.opt.vis_dir is None:
            self.VIS_DIR = './visualization/%s' % MODEL_NAME
        else:
            self.VIS_DIR = self.opt.vis_dir

        if not os.path.exists(self.VIS_DIR):
            os.makedirs(self.VIS_DIR)

        DEFAULT_CHECKPOINT_PATH = os.path.join(self.LOG_DIR, 'checkpoint.tar')
        self.CHECKPOINT_PATH = self.opt.checkpoint_path if self.opt.checkpoint_path is not None \
            else DEFAULT_CHECKPOINT_PATH
        print(f"log to {self.LOG_DIR}")

    def build_model_optimizer(self):

        model_dict = get_model_module(self.opt.model_dict)
        self.model = model_dict.PrimitiveNet(self.opt).cuda()
       
        total_parameters = self.model.parameters()

        parameter_count(self.model)
        
        if torch.cuda.device_count() > 1:
            print("Let's use %d GPUs!" % (torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)

        if self.opt.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                total_parameters,
                lr=self.opt.learning_rate,
                weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                total_parameters,
                lr=self.opt.learning_rate,
                momentum=self.opt.momentum,
                nesterov=True,
                weight_decay=self.opt.weight_decay)

        self.BASE_LEARNING_RATE = self.opt.learning_rate
        self.BN_DECAY_STEP = self.opt.bn_decay_step
        self.BN_DECAY_RATE = self.opt.bn_decay_rate
        self.LR_DECAY_STEPS = [
            int(x) for x in self.opt.lr_decay_steps.split(',')
        ]
        self.LR_DECAY_RATE = self.opt.lr_decay_rate
        self.load_checkpoint()
        
    def load_checkpoint(self):
        # Load checkpoint if any
        self.start_epoch = 0
        if self.CHECKPOINT_PATH is not None and os.path.isfile(
                self.CHECKPOINT_PATH) and not self.opt.not_load_model:
            print('load checkpoint path: %s' % self.CHECKPOINT_PATH)
            checkpoint = torch.load(self.CHECKPOINT_PATH)
            pretrained_dict = checkpoint['model_state_dict']
            model_dict = self.model.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            try:
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(e)
            self.start_epoch = checkpoint['epoch']
            print("Successfully Load Model with %d epoch..." %
                  self.start_epoch)

    def get_current_lr(self, epoch):
        lr = self.BASE_LEARNING_RATE
        for i, lr_decay_epoch in enumerate(self.LR_DECAY_STEPS):
            if epoch >= lr_decay_epoch:
                lr *= self.LR_DECAY_RATE
        self.TRAIN_VISUALIZER.log_scalars({'lr': lr}, self.epoch)
        return lr

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.get_current_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def build_dataloader(self):
        DATA_PATH = self.opt.data_path
        TRAIN_DATASET = self.opt.train_dataset
        TEST_DATASET = self.opt.test_dataset

        # Init datasets and dataloaders
        def my_worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        # Create dataset
        if self.opt.dataset == 'ABC':
            from dataloader.ABCDataset import ABCDataset
            Dataset = ABCDataset

        train_dataset = Dataset(DATA_PATH,
                                TRAIN_DATASET,
                                opt=self.opt,
                                skip=self.opt.train_skip,
                                fold=self.opt.train_fold)
        test_dataset = Dataset(DATA_PATH, TEST_DATASET, opt=self.opt, skip=self.opt.val_skip)

        num_workers = 0 if self.opt.debug else 4

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.opt.batch_size, \
                shuffle=True, num_workers=num_workers, worker_init_fn=my_worker_init_fn)

        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, \
                shuffle=False, num_workers=num_workers, worker_init_fn=my_worker_init_fn)

    def train_one_epoch(self):
        stat_dict = defaultdict(int)
        self.adjust_learning_rate(self.optimizer, self.epoch)

        self.model.train()
        data_time = time.time()
        iter_time_start = time.time()
        for batch_idx, batch_data_label in enumerate(self.train_dataloader):
            now = cuda_time()
            stat_dict['data_time'] += time.time() - data_time
            for key in batch_data_label:
                if not isinstance(batch_data_label[key], list):
                    batch_data_label[key] = batch_data_label[key].cuda()
            # Forward pass
            self.optimizer.zero_grad()

            with torch.autograd.set_detect_anomaly(True):
                total_loss, loss_dict = self.process_batch(batch_data_label)

                total_loss.backward()

            self.optimizer.step()
            stat_dict['step_time'] += time.time() - iter_time_start
            iter_time_start = time.time()

            # Accumulate statistics and print out
            for key in loss_dict:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += loss_dict[key].item()

            batch_interval = 50
            BATCH_SIZE = self.train_dataloader.batch_size
            if (batch_idx + 1) % batch_interval == 0:
                print('batch: %03d:' % (batch_idx + 1), end=' ')
                stat_dict['example/sec'] = BATCH_SIZE * 1.0 / (
                    stat_dict['step_time'] / batch_interval)
                self.TRAIN_VISUALIZER.log_scalars(
                    {
                        key: stat_dict[key] / batch_interval
                        for key in stat_dict
                    }, (self.epoch * len(self.train_dataloader) + batch_idx) *
                    BATCH_SIZE)

                print('example/sec: %.1f |' %
                      (BATCH_SIZE * 1.0 /
                       (stat_dict['step_time'] / batch_interval)),
                      end=' ')
                print('data/sec: %.1f |' %
                      (BATCH_SIZE * 1.0 /
                       (stat_dict['data_time'] / batch_interval)),
                      end=' ')
                print('data/step: %.1f |' %
                      ((stat_dict['data_time'] / stat_dict['step_time'])),
                      end=' ')
                for key in sorted(stat_dict.keys()):
                    if key not in ['step_time', 'data_time', 'example/sec']:
                        print('%s: %.3f |' %
                              (key, stat_dict[key] / batch_interval),
                              end=' ')
                    stat_dict[key] = 0
                print()

            data_time = time.time()

    def test_one_epoch(self):

        stat_dict = {}
        self.model.eval()
        cnt = 0

        for batch_idx, batch_data_label in enumerate(self.test_dataloader):
            if batch_idx % 200 == 0:
                print('Eval batch: %d' % (batch_idx))

            for key in batch_data_label:
                if not isinstance(batch_data_label[key], list):
                    batch_data_label[key] = batch_data_label[key].cuda()

            with torch.no_grad():
                total_loss, loss_dict = self.process_batch(batch_data_label,
                                                           postprocess=True)

            # Accumulate statistics and print out
            for key in loss_dict:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += loss_dict[key].item()
            cnt += len(batch_data_label['index'])

        for key in sorted(stat_dict.keys()):
            print('%s: %f' % (key, stat_dict[key] / cnt),
                  end=' ')
        
        print()
        # Log statistics
        BATCH_SIZE = self.test_dataloader.batch_size
        self.TEST_VISUALIZER.log_scalars(
            {key: stat_dict[key] / float(batch_idx + 1)
             for key in stat_dict},
            (self.epoch + 1) * len(self.test_dataloader) * BATCH_SIZE)
        
        miou = stat_dict['miou'] / (float(batch_idx + 1))
        return miou

    def train(self):
        
        max_miou = 0
        for epoch in range(self.start_epoch, self.opt.max_epoch):
            self.epoch = epoch
            print('**** EPOCH %03d ****' % (epoch))
            print('Current learning rate: %f' % (self.get_current_lr(epoch)))
            
            if self.opt.eval:
                self.test_one_epoch()
                break

            # Reset numpy seed.
            # REF: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed()
            self.train_one_epoch()
            # Eval every 10 epochs
            if epoch % self.opt.eval_interval == self.opt.eval_interval - 1:
                
                miou = self.test_one_epoch()
                # Save checkpoint
                save_dict = {
                    'epoch': epoch +
                    1,  # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    #'loss': test_loss,
                }

                if miou >= max_miou:
                    max_miou = miou
                    try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                        save_dict[
                            'model_state_dict'] = self.model.module.state_dict()
                    except:
                        save_dict['model_state_dict'] = self.model.state_dict()
                    torch.save(save_dict,
                               os.path.join(self.LOG_DIR, 'checkpoint.tar'))
                    
                if epoch % self.opt.save_interval == self.opt.save_interval - 1:
                    try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                        save_dict[
                            'model_state_dict'] = self.model.module.state_dict()
                    except:
                        save_dict['model_state_dict'] = self.model.state_dict()

                    torch.save(
                        save_dict,
                        os.path.join(self.LOG_DIR,
                                     'checkpoint_eval%d.tar' % epoch))


if __name__ == '__main__':
    FLAGS = build_option()
    trainer = Trainer(FLAGS)
    trainer.train()
