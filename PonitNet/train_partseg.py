import time
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils import label_to_onehot
from utils import get_seed, get_logger, get_work_dir
from data.shapenet_data import get_dataset, get_dataloader
from utils import AverageMeter
from losses import PartSegLoss
from pointnet import PointNetPartSeg

# ===== Set HyperParameter =====
# set seed
seed = 42

# set data
shapenet_folder = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
batch_size = 16
use_normals = True

# set train
base_lr = 1e-3
weight_decay = 1e-4
total_epochs = 150
# ==============================

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def validate(dataloader, model, criterion, total_batch, logger=None):

    model.eval()

    time_st = time.time()
    
    test_metrics = {}
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(50)]
    total_correct_class = [0 for _ in range(50)]    

    with paddle.no_grad():
        
        for batch_id, data in enumerate(dataloader):

            point_set = data[0]
            cls = data[1]
            seg = data[2]

            batchsize, npoints, _ = point_set.shape

            cls_onehot = label_to_onehot(cls, total_cls=16)
            seg_pred, trans_feat = model(point_set, cls_onehot)
            
            seg = seg.numpy()
            seg_pred = seg_pred.numpy()
            cur_pred_val = np.zeros([batchsize, npoints]).astype(np.int32)

            for i in range(batchsize):
                cat = seg_label_to_cat[seg[i, 0]]
                logits = seg_pred[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]    

            correct = np.sum(cur_pred_val == seg)
            total_correct += correct
            total_seen += (batchsize * npoints)            
            
            for l in range(50):
                total_seen_class[l] += np.sum(seg == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (seg == l)))

            for i in range(batchsize):
                segp = cur_pred_val[i, :]
                segl = seg[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            logger('- eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    val_time = time.time() - time_st
    return test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou'], val_time



def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          total_epochs,
          total_batch,
          logger=None):
    model.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()

    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        point_set = data[0]
        cls = data[1]
        seg = data[2]

        cls_onehot = label_to_onehot(cls, total_cls=16)
        seg_pred, trans_feat = model(point_set, cls_onehot)

        loss = criterion(seg_pred, seg, trans_feat)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        seg_pred = seg_pred.reshape([-1, 50])  # num part
        seg = seg.reshape([-1,1])

        seg_pred = F.softmax(seg_pred, axis=1)
        acc1 = paddle.metric.accuracy(seg_pred, seg)

        batch_size = point_set.shape[0]
        train_loss_meter.update(loss.numpy()[0], batch_size)
        train_acc_meter.update(acc1.numpy()[0], batch_size)

        report_freq = len(dataloader) // 20

        if logger and batch_id % report_freq == 0:
            logger.info(
                f"- Epoch[{epoch:>3d}/{total_epochs:>3d}], " +
                f"Step[{batch_id:>4d}/{total_batch:>4d}], " +
                f"Avg Loss: {train_loss_meter.avg:.4f}, " +
                f"Avg Acc: {train_acc_meter.avg:.4f}")

    train_time = time.time() - time_st

    return train_loss_meter.avg, train_acc_meter.avg, train_time


def main():
    
    # step 0: preparation
    get_seed(seed)
    work_dir = get_work_dir()
    logger = get_logger(work_dir)

	# step 1: create model
    model = PointNetPartSeg(use_normals=True)

	# setp 2: create train and val dataloader
    train_dataset = get_dataset(data='ShapeNet', file_folder=shapenet_folder, npoints=2048, mode='train', use_normals=True)
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, mode='train')
    val_dataset = get_dataset(data='ShapeNet', file_folder=shapenet_folder, npoints=2048, mode='val', use_normals=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, mode='val')

    # step 3: define criterion
    criterion = PartSegLoss()

    # step 4: define optimizer and lr_scheduler
    lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=base_lr,
                                                 step_size=20,
                                                 gamma=0.5)

    optimizer = paddle.optimizer.Adam(learning_rate=lr_scheduler,
                                      parameters=model.parameters(),
                                      weight_decay=weight_decay)


    # step 5: start training and validation
    for epoch in range(1, total_epochs+1):
    	# train
        logger.info(f"----- Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
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
        logger.info(f'----- Validation after Epoch: {epoch}')

        accuracy, class_avg_iou, inctance_avg_iou, val_time = validate(dataloader=val_dataloader,
									                                   model=model,
									                                   criterion=criterion,
									                                   total_batch=len(val_dataloader),
									                                   logger=logger)
        
        logger.info(f"==> Epoch[{epoch:>3d}/{total_epochs:>3d}], " +
                    f"Test Accuracy: {accuracy:.4f}, " +
                    f"Class avg mIOU: {class_avg_iou:.4f}, " +
                    f"Inctance avg mIOU: {inctance_avg_iou:.4f}, "
                    f"time: {val_time:.2f}")

        # save model(best)

if __name__ == "__main__":
    main()










