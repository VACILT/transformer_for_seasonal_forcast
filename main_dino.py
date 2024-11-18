# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


import utils
import vision_transformer as vits



def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] ,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")                                                  # modified by Sina
    parser.add_argument('--number_of_patches', default=156, type=int, help="""Size in number
        of input patches - default 156 """)                                                            # modified by Sina


    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")


    # Misc
    parser.add_argument('--data_path', default='/work/bb1438/b381522/VIT/transformer_for_seasonal_forcast/Datasets/', type=str,
        help='Please specify path to the training data.')           # modified by Sina
    parser.add_argument('--output_dir', default="/work/bb1438/b381522/VIT/transformer_for_seasonal_forcast/Codes/outputs/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_model(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    dataset = utils.ChunkedH5Dataset(args.data_path)                     # added by Sina, load the specific dataset
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    dataset_val = utils.ValidationH5Dataset(args.data_path)                     # added by Sina, load the specific dataset
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )


    print(f"Data loaded: there are {len(dataset)} data points.")       # modified by Sina

    # ============ building the Model ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        Model = vits.__dict__[args.arch](
            num_patches=args.number_of_patches,  # Modified by Sina
            drop_path_rate=args.drop_path_rate,  # stochastic depth, 
        )
        embed_dim = Model.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # move networks to gpu
    Model = Model.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(Model):
        Model = nn.SyncBatchNorm.convert_sync_batchnorm(Model)

    Model = nn.parallel.DistributedDataParallel(Model, device_ids=[args.gpu])
    
    
    print(f"The model is built: It is a {args.arch} network.")

    # ============ preparing loss ... ============
    model_loss = MODELLoss().cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(Model)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        Model=Model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        model_loss=model_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting Model training !")
    for epoch in range(start_epoch, args.epochs):
        
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(Model, model_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule,epoch,
                    fp16_scaler, args)
        
        
        # ============ validation after each epoch ============
        val_stats = validate(
            Model, model_loss, data_loader_val, fp16_scaler
        )
        Model.train()
        # ============ writing logs ... ============
        save_dict = {
            'Model': Model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'model_loss': model_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

        # Combine training and validation stats
        log_stats = {
        **{f'train_{k}': v for k, v in train_stats.items()},
        **{f'val_{k}': v for k, v in val_stats.items()},
        'epoch': epoch
        }

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
 


def train_one_epoch(Model, model_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (input_data, output_data) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move data to gpu
        input_data = [im.cuda(non_blocking=True) for im in input_data]
        output_data = [im.cuda(non_blocking=True) for im in output_data]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None): 
            model_output = Model(input_data)
            loss = model_loss(output_data, model_output)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(Model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, Model,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(Model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, Model,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(Model, model_loss, data_loader_val, fp16_scaler):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'
    Model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for input_data_val, output_data_val in metric_logger.log_every(data_loader_val, 10, header):
            input_data_val = [im.cuda(non_blocking=True) for im in input_data_val]
            output_data_val = [im.cuda(non_blocking=True) for im in output_data_val]

            if fp16_scaler is None:
                model_output = Model(input_data_val)
                loss = model_loss(output_data_val, model_output)
            else:
                with torch.cuda.amp.autocast(True):
                    model_output = Model(input_data_val)
                    loss = model_loss(output_data_val, model_output)

            metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print("Validation Loss:", metric_logger.loss.global_avg)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



class MODELLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean') 
        
    def forward(self, output_data, model_output):
        """
        Compute the Mean Squared Error (MSE) between output_data and model_output.
        """
        # Ensure the tensors are on the same device
        assert output_data.device == model_output.device, "Tensors must be on the same device"

        total_loss = self.mse_loss(model_output, output_data)

        return total_loss
    

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)





if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_model(args)
