import datetime
import os
import sys
import argparse
import logging
import matplotlib.pyplot as plt

import cv2
import torch
import torch.utils.data
import torch.optim as optim

from torchsummary import summary
import tensorboardX

from utils.visualisation.gridshow import gridshow
from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')
    parser.add_argument('--network', type=str, default='ggcnn', help='Network Name in .models')
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')
    
    return parser.parse_args()


def validate(net, device, val_data, batches_per_epoch):
    net.eval()
    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {},
        'angle_error': 0,  
        'width_error': 0,  
        'confidence': 0  
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']
                results['loss'] += loss.item() / ld
                for ln, l in lossd['losses'].items():
                    results['losses'][ln] = results['losses'].get(ln, 0) + l.item() / ld

                q_out, ang_out, w_out = post_process_output(
                    lossd['pred']['pos'],
                    lossd['pred']['cos'],
                    lossd['pred']['sin'],
                    lossd['pred']['width']
                )

                # 计算误差
                gt_bb = val_data.dataset.get_gtbb(didx, rot, zoom_factor)
                results['angle_error'] += torch.mean(torch.abs(ang_out - gt_bb.angle)) / ld
                results['width_error'] += torch.mean(torch.abs(w_out - gt_bb.width)) / ld
                results['confidence'] += torch.mean(q_out).item() / ld  

                s = evaluation.calculate_iou_match(q_out, ang_out, gt_bb, no_grasps=1, grasp_width=w_out)
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

    return results


def run():
    args = parse_args()

    # 训练曲线保存路径
    save_folder = "output/training_results"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, datetime.datetime.now().strftime('%y%m%d_%H%M')))

    # 加载数据集
    logging.info('Loading Dataset...')
    Dataset = get_dataset(args.dataset)
    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, include_depth=args.use_depth, include_rgb=args.use_rgb)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    net = get_network(args.network)(input_channels=input_channels).to(torch.device("cuda:0"))
    optimizer = optim.Adam(net.parameters())

    # 训练曲线数据
    train_losses, val_losses, iou_scores, grasp_success_rates = [], [], [], []
    angle_errors, width_errors, grasp_confidences = [], [], []

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info(f'Beginning Epoch {epoch}')
        train_results = validate(net, torch.device("cuda:0"), train_data, args.batches_per_epoch)
        train_losses.append(train_results['loss'])

        logging.info('Validating...')
        test_results = validate(net, torch.device("cuda:0"), val_data, args.val_batches)

        val_losses.append(test_results['loss'])
        iou_scores.append(test_results['correct'] / (test_results['correct'] + test_results['failed']))
        grasp_success_rates.append(iou_scores[-1])
        angle_errors.append(test_results['angle_error'])
        width_errors.append(test_results['width_error'])
        grasp_confidences.append(test_results['confidence'])

    # 绘制曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1), plt.plot(train_losses, label="Train Loss"), plt.plot(val_losses, label="Validation Loss"), plt.legend()
    plt.subplot(2, 3, 2), plt.plot(iou_scores, label="IOU Score"), plt.legend()
    plt.subplot(2, 3, 3), plt.plot(grasp_success_rates, label="Grasp Success Rate"), plt.legend()
    plt.subplot(2, 3, 4), plt.plot(angle_errors, label="Angle Error"), plt.legend()
    plt.subplot(2, 3, 5), plt.plot(width_errors, label="Width Error"), plt.legend()
    plt.subplot(2, 3, 6), plt.plot(grasp_confidences, label="Grasp Confidence"), plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "training_results.png"))  # 按你的要求保存训练曲线
    plt.show(block=False)  # 防止阻塞训练

if __name__ == '__main__':
    run()