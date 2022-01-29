import time
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils import get_seed, get_logger, get_work_dir
from modelnet_data import get_dataset, get_dataloader
from utils import AverageMeter
from losses import ClsLoss
from pointnet import PointNetClassifier


# other
seed = 42

# data
model40_folder = 'modelnet40_normal_resampled'
batch_size = 32
use_normals = True

# train
base_lr = 1e-3
weight_decay = 1e-4
total_epochs = 200


def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          total_epochs,
          total_batch,
          report_freq=30,
          logger=None):
    model.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()

    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        point_set = data[0]
        label = data[1]

        pred, trans_input, trans_feat = model(point_set)
        loss = criterion(pred, label, trans_feat)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        pred = F.softmax(pred, axis=1)
        acc1 = paddle.metric.accuracy(pred, label)

        batch_size = point.shape[0]
        train_loss_meter.update(loss.numpy()[0], batch_size)
        train_acc_meter.update(acc1.numpy()[0], batch_size)


        if logger and batch_id % report_freq == 0:
            logger.info(
                f"Epoch[{epoch:>3d}/{total_epochs:>3d}], " +
                f"Step[{batch_id:>4d}/{total_batch:>4d}], " +
                f"Avg Loss: {train_loss_meter.avg:.4f}, " +
                f"Avg Acc: {train_acc_meter.avg:.4f}")

    train_time = time.time() - time_st

    return train_loss_meter.avg, train_acc_meter.avg, train_time

def validate(dataloader, model, criterion, total_batch, report_freq=40, logger=None):

    model.eval()
    val_loss_meter = AverageMeter()
    val_acc1_meter = AverageMeter()

    time_st = time.time()

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            point = data[0]
            label = data[1]

            pred, trans_input, trans_feat = model(point)
            loss = criterion(pred, label, trans_feat)

            pred = F.softmax(pred, axis=1)
            acc1 = paddle.metric.accuracy(pred, label)

            batch_size = point.shape[0]

            val_loss_meter.update(loss.numpy()[0], batch_size)
            val_acc1_meter.update(acc1.numpy()[0], batch_size)

            if logger and batch_id % report_freq == 0:
                logger.info(
                    f"Val Step[{batch_id:>4d}/{total_batch:>4d}], " +
                    f"Avg Loss: {val_loss_meter.avg:.4f}, " +
                    f"Avg Acc@1: {val_acc1_meter.avg:.4f}")

    val_time = time.time() - time_st
    return val_loss_meter.avg, val_acc1_meter.avg, val_time



def main():
    
    # step 0: preparation
    get_seed(seed)
    work_dir = get_work_dir()
    logger = get_logger(work_dir)

	# step 1: create model
    model = PointNetClassifier(use_normals=True)

	# setp 2: create train and val dataloader
    train_dataset = get_dataset(data='ModelNet40', file_folder=modelnet40_folder, npoints=1024, mode='train', use_normals=True)
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, mode='train')
    val_dataset = get_dataset(data='ModelNet40', file_folder=modelnet40_folder,npoints=1024, mode='val', use_normals=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, mode='val')

    # step 3: define criterion
    criterion = ClsLoss()

    # step 4: define optimizer and lr_scheduler
    lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=base_lr,
                             step_size=20,
                             gamma=0.7)

    optimizer = paddle.optimizer.Adam(learning_rate=lr_scheduler,
                                      parameters=model.parameters(),
                                      weight_decay=weight_decay)

    # step 5: validation (eval mode)

    # step 6: start training and validation (train mode)
    for epoch in range(1, total_epochs+1):
    	# train
        logger.info(f"----- Now training epoch {epoch}. LR={optimizer.get_lr():.6f} -----")
        train_loss, train_acc, train_time = train(dataloader=train_dataloader,
                                                  model=model,
                                                  criterion=criterion,
                                                  optimizer=optimizer,
                                                  epoch=epoch,
                                                  total_epochs=total_epochs,
                                                  total_batch=len(train_dataloader),
                                                  logger=logger)
        lr_scheduler.step()
        logger.info(f"==> Epoch[{epoch:>3d}/{total_epochs:>3d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"Train Acc: {train_acc:.4f}, " +
                    f"time: {train_time:.2f}")

        # validation
        logger.info(f'----- Validation after Epoch: {epoch} -----')
        val_loss, val_acc1, val_time = validate(dataloader=val_dataloader,
									            model=model,
									            criterion=criterion,
									            total_batch=len(val_dataloader),
									            logger=logger)
        
        logger.info(f"==> Epoch[{epoch:>3d}/{total_epochs:>3d}], " +
                    f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Acc@1: {val_acc1:.4f}, " +
                    f"time: {val_time:.2f}")

        # model save

if __name__ == "__main__":
    main()










