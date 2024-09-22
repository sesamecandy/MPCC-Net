# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir
import time
import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    
    # 伴随测试
    model_path = cfg.OUTPUT_DIR
    total_epoch = cfg.SOLVER.MAX_EPOCHS
    interval_epoch = cfg.SOLVER.CHECKPOINT_PERIOD
    index_list = [i for i in range(0, total_epoch, interval_epoch)]
    index_list = index_list[1:]
    
    for idx in index_list:
        model_name = f"resnet50_ibn_a_model_{idx}.pth"
        load_path = os.path.join(model_path, model_name)
        
        while not os.path.exists(load_path):
            time.sleep(10)
        
        model = torch.load(load_path)
        inference(cfg, model, val_loader, num_query)

if __name__ == '__main__':
    main()
