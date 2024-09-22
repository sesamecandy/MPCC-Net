# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils.reid_metric import R1_mAP
from .consistency_loss import Consistent_compare
import math

global ITER
ITER = 0


def visualize_attention_map(score_map_img, score_map_aug, score_map_flip, save_path='/home/lucky7/heatmap_train'):
    """
    Visualize and save the attention maps.

    Args:
        score_map_img (torch.Tensor): The attention map for original image.
        score_map_aug (torch.Tensor): The attention map for augmented image.
        score_map_flip (torch.Tensor): The attention map for flipped image.
        save_path (str): The directory to save the attention maps.
    """
    # Ensure the save directory exists
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Assuming the tensors are in [64, 1, 24, 8] shape
    batch_size = score_map_img.size(0)

    for i in range(batch_size):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        score_map_img_np = score_map_img[i, 0, :, :].detach().cpu().numpy()
        score_map_aug_np = score_map_aug[i, 0, :, :].detach().cpu().numpy()
        score_map_flip_np = score_map_flip[i, 0, :, :].detach().cpu().numpy()

        axs[0].imshow(score_map_img_np, cmap='hot', interpolation='nearest')
        axs[0].set_title(f'Original Image {i + 1} Attention Map')
        axs[1].imshow(score_map_aug_np, cmap='hot', interpolation='nearest')
        axs[1].set_title(f'Augmented Image {i + 1} Attention Map')
        axs[2].imshow(score_map_flip_np, cmap='hot', interpolation='nearest')
        axs[2].set_title(f'Flipped Image {i + 1} Attention Map')

        plt.tight_layout()
        file_path = os.path.join(save_path, f'attention_map_{i + 1}.png')
        plt.savefig(file_path)
        plt.close(fig)

def create_supervised_trainer(model, optimizer, loss_fn, loss_attention_cri, with_closs,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    score_maps = []
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        # img_aug = F.upsample_bilinear(img, [336, 112])#288,96     #增强之后的图像
        score_img, feat_img, score_map_img, base_map_img = model(img, target)  #分数 特征图 热图 压缩到单通道的图
        # score_aug, feat_aug, score_map_aug, base_map_aug = model(img_aug, target)
        # img_flip = img_aug.flip(-1)
        # score_flip, feat_flip, score_map_flip, base_map_flip = model(img_flip, target)

        # score_maps.clear()
        # score_maps.append((score_map_img, score_map_aug, score_map_flip))
        
        # loss = loss_fn(score_img, feat_img, target) + loss_fn(score_aug, feat_aug, target) + loss_fn(score_flip, feat_flip, target)
        loss = loss_fn(score_img[0], feat_img[0], target) + loss_fn(score_img[1], feat_img[1], target) + loss_fn(score_img[2],
                                                                                                     feat_img[2], target)
        loss /= 3
        
        #  如果使用一致性loss，则计算下面的操作
        if with_closs == "on":
            img_w, aug_w = score_map_img[0].shape[-1], score_map_img[1].shape[-1]
            GCD = math.gcd(img_w, aug_w)
            LCM = img_w * aug_w // GCD
            #  print(LCM)
            attention_loss_aug = Consistent_compare(scoremap1=score_map_img[1], scoremap2=score_map_img[0], loss_attenion_cri=loss_attention_cri, mode='scaling', size=[LCM*3, LCM])
            attention_loss_flip = Consistent_compare(scoremap1=score_map_img[2].flip(-1), scoremap2=score_map_img[0], loss_attenion_cri=loss_attention_cri, mode='scaling', size=[LCM*3, LCM])
            attention_loss_base_img = Consistent_compare(scoremap1=base_map_img[0], scoremap2=score_map_img[0], loss_attenion_cri=loss_attention_cri)
            attention_loss_base_aug = Consistent_compare(scoremap1=base_map_img[1], scoremap2=score_map_img[1], loss_attenion_cri=loss_attention_cri)
            attention_loss_base_flip = Consistent_compare(scoremap1=base_map_img[2], scoremap2=score_map_img[2], loss_attenion_cri=loss_attention_cri)
            base_map_loss = (attention_loss_base_img + attention_loss_base_aug + attention_loss_base_flip) / 3
            loss = loss + 0.01 * (attention_loss_aug + attention_loss_flip+ base_map_loss) / 3
        
        
        loss.backward()
        optimizer.step()
        # compute acc
        acc = ((score_img[0].max(1)[1] == target).float().mean()
            + (score_img[1].max(1)[1] == target).float().mean() \
                + (score_img[2].max(1)[1] == target).float().mean()) / 3
        return loss.item(), acc.item()

    return Engine(_update)#, score_maps


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        loss_attention_cri,
        with_closs,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    if with_closs == "on":
        logger.info("======> use consistency loss!!! <======")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, loss_attention_cri, with_closs, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=cfg.SOLVER.MAX_EPOCHS, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

            # for score_map_img, score_map_aug, score_map_flip in score_maps:
            #     visualize_attention_map(score_map_img, score_map_aug, score_map_flip)

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=cfg.SOLVER.MAX_EPOCHS, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)
