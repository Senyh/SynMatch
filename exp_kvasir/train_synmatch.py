import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import time
import argparse
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize
from torchvision.transforms import ToPILImage
from exp_kvasir.data.dataset import KvasirDataset
from models import networks
from wheels.loss_functions import DSCLossH
from wheels.logger import logger as logging
from utils import ensure_dir
from wheels.mask_generator import BoxMaskGenerator, AddMaskParamsToBatch, SegCollate
from wheels.torch_utils import seed_torch
from wheels.model_init import init_weight
from wheels.activation_extraction import ActivationExtraction
from wheels.colorspace_transform import rgb_to_ycbcr_batch, ycbcr_to_rgb_batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--seed', type=int, default=2025, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/SynMatch', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['UNet'], help='segmentation backbone')
    parser.add_argument('--ssl', type=bool, default=False, help='ssl or bsl')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--labeled_percentage', type=float, default=.05, help='the percentage of labeled data')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--log_freq', type=float, default=10, help='logging frequency of metrics accord to the current iteration')
    parser.add_argument('--save_freq', type=float, default=10, help='saving frequency of model weights accord to the current epoch')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    val_set = KvasirDataset(image_path=args.data_path, stage='val', image_size=args.image_size, is_augmentation=False)
    labeled_train_set = KvasirDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=True, percentage=args.labeled_percentage, ssl=args.ssl)
    unlabeled_train_set = KvasirDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=False, percentage=args.labeled_percentage)
    train_set = ConcatDataset([labeled_train_set, unlabeled_train_set])

    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    print('before: ', len(train_set), len(labeled_train_set), len(unlabeled_train_set), len(val_set))
    labeled_ratio = len(train_set) // len(labeled_train_set)
    labeled_train_set = ConcatDataset([labeled_train_set for i in range(labeled_ratio)])
    labeled_train_set = ConcatDataset([labeled_train_set,
                                       Subset(labeled_train_set, range(len(train_set) - len(labeled_train_set)))])
    print('after: ', len(train_set), len(labeled_train_set), len(unlabeled_train_set), len(val_set))
    assert len(labeled_train_set) == len(train_set)
    train_labeled_dataloder = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    train_unlabeled_dataloder = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataloder = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    lmask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5),
                                        n_boxes=3,
                                        random_aspect_ratio=True,
                                        prop_by_area=True,
                                        within_bounds=True,
                                        invert=True)

    add_lmask_params_to_batch = AddMaskParamsToBatch(lmask_generator)
    lmask_collate_fn = SegCollate(batch_aug_fn=add_lmask_params_to_batch)
    aux_labeled_dataloader = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=lmask_collate_fn)
    mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5),
                                        n_boxes=3,
                                        random_aspect_ratio=True,
                                        prop_by_area=True,
                                        within_bounds=True,
                                        invert=True)

    add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
    aux_unlabeled_dataloader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=mask_collate_fn)
    return train_labeled_dataloder, train_unlabeled_dataloder, val_dataloder, aux_unlabeled_dataloader, aux_labeled_dataloader


def main(is_debug=False):
    args = get_args()
    seed_torch(args.seed)
    # Project Saving Path
    if args.ssl:
        project_path = args.project + '_{}_label_{}_SSL/'.format(args.backbone, args.labeled_percentage)
    else:
        project_path = args.project + '_{}_label_{}/'.format(args.backbone, args.labeled_percentage)
    ensure_dir(project_path)
    save_path = project_path + 'weights/'
    ensure_dir(save_path)

    # Tensorboard & Statistics Results & Logger
    tb_dir = project_path + '/tensorboard{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    writer = SummaryWriter(tb_dir)
    metrics = EasyDict()
    metrics.train_loss = []
    metrics.train_s_loss = []
    metrics.train_u_loss = []
    metrics.val_loss = []
    logger = logging(project_path + 'train_val.log')
    logger.info('PyTorch Version {}\n Experiment{}'.format(torch.__version__, project_path))

    # Load Data
    train_labeled_dataloader, train_unlabeled_dataloader, val_dataloader, aux_unlabeled_dataloader, aux_labeled_dataloader = get_data(args=args)
    iters = len(train_labeled_dataloader)
    val_iters = len(val_dataloader)

    # Load Model & EMA
    student = networks.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    init_weight(student.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='leaky_relu')
    activation_extractor = ActivationExtraction(model=student, texture_layer=student.net.backbone.in_conv, shape_layer=student.net.classifier.up4).to(device)
    best_model_wts = copy.deepcopy(student.state_dict())
    logger.info('#parameters: {}'.format(sum(param.numel() for param in student.parameters())))
    best_epoch = 0
    best_loss = 100
    conf_threshold = 0.95  # the confidence threshold for pixel-level pseudo label selection

    # Criterion & Optimizer & LR Schedule
    criterion = DSCLossH(num_classes=args.num_classes, device=device)
    criterion_u = DSCLossH(num_classes=args.num_classes, device=device)
    optimizer = optim.AdamW(student.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    norm = Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
    denorm = Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    # Train
    since = time.time()
    logger.info('start training')
    for epoch in range(1, args.num_epochs + 1):
        epoch_metrics = EasyDict()
        epoch_metrics.train_loss = []
        epoch_metrics.train_s_loss = []
        epoch_metrics.train_d_loss = []
        epoch_metrics.train_u_loss = []
        if is_debug:
            pbar = range(10)
        else:
            pbar = range(iters)
        iter_train_labeled_dataloader = iter(train_labeled_dataloader)
        iter_train_unlabeled_dataloader = iter(train_unlabeled_dataloader)
        iter_aux_unlabeled_loader = iter(aux_unlabeled_dataloader)
        iter_aux_labeled_loader = iter(aux_labeled_dataloader)
        ############################
        # Train
        ############################
        student.train()
        for idx in pbar:
            # sparse labeled data
            limage, label, limageA1, limageA2 = next(iter_train_labeled_dataloader)
            limage, label = limage.to(device), label.to(device)
            limageA1, limageA2 = limageA1.to(device), limageA2.to(device)
            if args.ssl:
                barely_mask = torch.ones_like(label.squeeze(1))
            else:
                barely_mask = label.squeeze(1) >= 0
                label[label < 0] = 0
            # unlabel data
            uimage, _, uimageA1, uimageA2 = next(iter_train_unlabeled_dataloader)
            uimage, uimageA1, uimageA2 = uimage.to(device), uimageA1.to(device), uimageA2.to(device)
            # auxiliary data
            uaimage, _, uaimageA1, uaimageA2, uamask = next(iter_aux_unlabeled_loader)
            uaimage = uaimage.to(device)
            uaimageA1, uaimageA2, uamask = uaimageA1.to(device), uaimageA2.to(device), uamask.to(device).long()
            laimage, lalabel, laimageA1, laimageA2, lamask = next(iter_aux_labeled_loader)
            laimage, lalabel = laimage.to(device), lalabel.to(device)
            if args.ssl:
                barely_amask = torch.ones_like(lalabel.squeeze(1))
            else:
                barely_amask = lalabel.squeeze(1) >= 0
                lalabel[lalabel < 0] = 0
            laimageA1, laimageA2, lamask = laimageA1.to(device), laimageA2.to(device), lamask.to(device).long()
            

            optimizer.zero_grad()

            # pseudo labelling
            with torch.no_grad():
                pred_uo = student(uimage)
                pred_uo_logits = pred_uo['out']
                pred_uo_probs = torch.softmax(pred_uo_logits, dim=1) # 8 4 256 256
                pred_uo_pseudo = torch.argmax(pred_uo_probs, dim=1).detach() # 8 256 256
                pred_uo_conf = pred_uo_probs.max(dim=1)[0].clone()
                
                pred_ua = student(uaimage)
                pred_ua_logits = pred_ua['out']
                pred_ua_probs = torch.softmax(pred_ua_logits, dim=1)
                pred_ua_pseudo = torch.argmax(pred_ua_probs, dim=1).clone()
                pred_ua_conf = pred_ua_probs.max(dim=1)[0].clone()
                    
                # cut mix
                if torch.rand(1) < .5:
                    limageA1[lamask.expand_as(uimageA1) == 1] = laimageA1[lamask.expand_as(limageA1) == 1]
                    label[lamask == 1] = lalabel[lamask == 1]
                    barely_mask[lamask.squeeze(1) == 1] = barely_amask[lamask.squeeze(1) == 1]
                
                # cut mix
                if torch.rand(1) < .5:
                    uimageA1[uamask.expand_as(uimageA1) == 1] = uaimageA1[uamask.expand_as(uimageA1) == 1]
                    pred_uo_pseudo[uamask.squeeze(1) == 1] = pred_ua_pseudo[uamask.squeeze(1) == 1]
                    pred_uo_conf[uamask.squeeze(1) == 1] = pred_ua_conf[uamask.squeeze(1) == 1]
                
            # the forward pass of unlabeled images    
            pred_uA1 = activation_extractor(uimageA1)
            pred_uA1_logits = pred_uA1['out']
            # unlabeled image synthesis
            with torch.no_grad():
                texture_map_u = activation_extractor.texture_activation[0]
                shape_map_u = activation_extractor.shape_activation[0]
                texture_map_u = torch.mean(texture_map_u, dim=1, keepdim=True)
                texture_map_u = torch.clamp(texture_map_u, 0, 1)
                shape_map_u = torch.mean(shape_map_u, dim=1, keepdim=True)
                shape_map_u = torch.clamp(shape_map_u, 0, 1)
                lam = np.random.beta(1., 1.)
                uimageA2Y = lam * texture_map_u + (1 - lam) * shape_map_u 
                
                p5_uimageA2Y = torch.quantile(uimageA2Y, 0.01)
                p95_uimageA2Y = torch.quantile(uimageA2Y, 0.99)
                uimageA2Y = torch.clamp(uimageA2Y, p5_uimageA2Y, p95_uimageA2Y)
                uimageA2 = denorm(uimageA2)
                uimageA2YCbCr = rgb_to_ycbcr_batch(uimageA2)
                uimageA2Yo = uimageA2YCbCr[:, 0].unsqueeze(1)
                uimageA2Y[uimageA2Yo <= 0] = 0
                uimageA2Y = uimageA2Y.detach()
                uimageA2YCbCr[:, 0] = uimageA2Y[:, 0]
                uimageA2 = ycbcr_to_rgb_batch(uimageA2YCbCr).detach()
                uimageA2 = norm(uimageA2)

            # the forward pass of synthesized unlabeled images
            pred_uA2 = student(uimageA2)
            pred_uA2_logits = pred_uA2['out']
            
            # the forward pass of labeled images
            pred_lA1 = student(limageA1)
            pred_lA1_logits = pred_lA1['out']
            
            # the supervised loss
            loss_s = criterion(pred_lA1_logits, label.squeeze(1).long(), pixel_mask=barely_mask.float()) 
            # the unsupervised loss
            loss_u = (criterion_u(pred_uA1_logits, pred_uo_pseudo, pixel_mask=(pred_uo_conf >= conf_threshold).float()) \
                + criterion_u(pred_uA2_logits, pred_uo_pseudo, pixel_mask=(pred_uo_conf >= conf_threshold).float())) / 2.

            loss = (loss_s + loss_u) / 2.

            loss.backward()
            optimizer.step()

            include_rate = (pred_uo_conf >= conf_threshold).sum() / pred_uo_conf.numel()
            writer.add_scalar('train_loss_s', loss_s.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss_u', loss_u.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('include_rate', include_rate, idx + len(pbar) * (epoch-1))
            if idx % args.log_freq == 0:
                logger.info("Train: Epoch/Epochs {}/{}, iter/iters {}/{}, "
                            "loss {:.3f}, loss_s {:.3f}, loss_u {:.3f}, included_rate {:.3f}"
                            .format(epoch, args.num_epochs, idx, len(pbar), 
                                    loss.item(), loss_s.item(), loss_u.item(), include_rate))
            epoch_metrics.train_loss.append(loss.item())
            epoch_metrics.train_s_loss.append(loss_s.item())
            epoch_metrics.train_u_loss.append(loss_u.item())
        metrics.train_loss.append(np.mean(epoch_metrics.train_loss))
        metrics.train_s_loss.append(np.mean(epoch_metrics.train_s_loss))
        metrics.train_u_loss.append(np.mean(epoch_metrics.train_u_loss))

        ############################
        # Validation
        ############################
        epoch_metrics.val_loss = []
        iter_val_dataloader = iter(val_dataloader)
        if is_debug:
            val_pbar = range(10)
        else:
            val_pbar = range(val_iters)
        student.eval()
        with torch.no_grad():
            for idx in val_pbar:
                image, label = next(iter_val_dataloader)
                image, label = image.to(device), label.to(device)
                pred = student(image)['out']
                loss = criterion(pred, label.squeeze(1).long())
                writer.add_scalar('val_loss', loss.item(), idx + len(val_pbar) * (epoch-1))
                if idx % args.log_freq == 0:
                    logger.info("Val: Epoch/Epochs {}/{}\t"
                                "iter/iters {}/{}\t"
                                "loss {:.3f}".format(epoch, args.num_epochs, idx, len(val_pbar),
                                                     loss.item()))
                epoch_metrics.val_loss.append(loss.item())
        metrics.val_loss.append(np.mean(epoch_metrics.val_loss))

        # Save Model
        if np.mean(epoch_metrics.val_loss) <= best_loss:
            best_model_wts = copy.deepcopy(student.state_dict())
            best_epoch = epoch
            best_loss = np.mean(epoch_metrics.val_loss)
            torch.save(best_model_wts, save_path + 'best.pth'.format(best_epoch))
        torch.save(student.state_dict(), save_path + 'last.pth'.format(best_epoch))
        logger.info("Average: Epoch/Epoches {}/{}, "
                    "train epoch loss {:.3f}, "
                    "val epoch loss {:.3f}, "
                    "best loss {:.3f} at {}\n".format(epoch, args.num_epochs, np.mean(epoch_metrics.train_loss),
                                                     np.mean(epoch_metrics.val_loss), best_loss, best_epoch))
    ############################
    # Save Metrics
    ############################
    data_frame = pd.DataFrame(
        data={'loss': metrics.train_loss,
              'loss_s': metrics.train_s_loss,
              'loss_u': metrics.train_u_loss,
              'val_loss': metrics.val_loss},
        index=range(1, args.num_epochs + 1))
    data_frame.to_csv(project_path + 'train_val_loss.csv', index_label='Epoch')
    plt.figure()
    plt.title("Loss During Training and Validating")
    plt.plot(metrics.train_loss, label="Train")
    plt.plot(metrics.val_loss, label="Val")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(project_path + 'train_val_loss.png')

    print(project_path)
    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('TRAINING FINISHED!')


if __name__ == '__main__':
    main()