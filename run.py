import argparse
import warnings
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
import matplotlib.pyplot as plt
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from torchvision import transforms
from torchvision.transforms import ToPILImage
import wandb
warnings.filterwarnings("ignore", category=UserWarning) 

model_names = sorted(name for name in models.__dict__
                        if name.islower() and not name.startswith("__")
                        and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10','cardiac_data'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--save_dir', default='checkpoints/', type = str, help= "specify the folder where to store the checkpoints")

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        print("Training with GPU")
        cudnn.deterministic = True
        cudnn.benchmark = True
        args.gpu_index = 0
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
        print("Training with CPU")

    dataset = ContrastiveLearningDataset(args.data,args.dataset_name)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views) # this has to  e tuple of list but its just returning list
    
    
    print(f"Number of samples in dataset: {len(train_dataset)}")

    # print("Type of train_dataset after all the transformations before dataloader",type(train_dataset))
    # #sample data has to be the tuple
   
    # sample_data = train_dataset[1]
    # print(sample_data)
    
    # img1=sample_data[0] #this has to be list 
    # img2=sample_data[1]
    
    # print("Sample data type:", type(sample_data))
    # print("Number of elements in sample data:", len(sample_data))
   
    # print("Type of img1:", type(img1))
    # print("Length of img1:", len(img1))
    # print("Contents of view1:", img1[0])
    # print("contents of view2",img1[1])
    

    # print(" Unique elements if img[1] tensor", torch.unique(img1[1]))

    # img1_np=img1[0]
    # img1_np = img1_np.numpy()
    # # # img1_np = (img1_np - img1_np.min()) / (img1_np.max() - img1_np.min())  # Normalize to [0, 1]
    # img2_np=img2[0]
    # img2_np = img2_np.numpy()
    # img2_np = (img2_np - img2_np.min()) / (img2_np.max() - img2_np.min())  # Normalize to [0, 1]

#     #Transpose the image dimensions from [C, H, W] to [H, W, C] for plotting
#     img1_np = np.transpose(img1, (1, 2, 0))
#     img2_np = np.transpose(img2, (1, 2, 0))

# # Display the image
#     plt.imshow(img1_np)
#     plt.title("View 2")
#     plt.savefig("/home/bk7944/segmentation/SimCLR/view1.jpg")
#     plt.imshow(img2_np)
#     plt.title("View 2")
#     plt.savefig("/home/bk7944/segmentation/SimCLR/view2.jpg")
    
    
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    
    # print("Number of batches",len(train_loader))
    # print("Type of dataset after dataloader", type(train_loader))
    
    print('===============>.....Training model.... <================')
  

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                            last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
