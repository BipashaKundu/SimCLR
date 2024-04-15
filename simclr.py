import logging
import os
import sys
from optparse import OptionParser
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import wandb

torch.manual_seed(0)
wandb.login()

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        # self.writer = SummaryWriter()
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        # self.writer = SummaryWriter(log_dir=self.args.save_dir)
        logging.basicConfig(filename=os.path.join(self.args.save_dir, 'training.log'), level=logging.DEBUG)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        wandb.init(project='Simclr Training', config=vars(self.args))
    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)



        # select only the negatives the negatives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)
    
        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Disable Cuda?: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):


            for images, _ in tqdm(train_loader): 

            
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                


                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    wandb.log({'loss': loss.item(), 'acc/top1': top1[0], 'acc/top5': top5[0], 'step': n_iter, 'epoch': epoch_counter})
                    # self.writer.add_scalar('loss', loss, global_step=n_iter)
                    # self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    # self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    # self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
               
            # warmup for the first 10 
            if epoch_counter >= 10:
                self.scheduler.step()
            wandb.log({'epoch_loss': loss.item(), 'epoch': epoch_counter, 'top1_epoch_accuracy': top1[0]})
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
        
            

                
            checkpoint_name = f'checkpoint_epoch_{epoch_counter:04d}.pth.tar'   
            # checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
            save_checkpoint_path = os.path.join(self.args.save_dir, checkpoint_name)
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=save_checkpoint_path)
        # wandb.log({"checkpoint_path": save_checkpoint_path,"loss": loss.item()})
            wandb.log({'checkpoint_path': save_checkpoint_path})
        logging.info("Training has finished.")
        logging.info(f"Checkpoints saved in directory {self.args.save_dir}")
        # save model checkpoints
        wandb.finish()
            

