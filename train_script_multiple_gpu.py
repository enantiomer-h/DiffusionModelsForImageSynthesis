#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified.
# @author: eschweiler
"Train the image-synthesis model."


from argparse import ArgumentParser

import numpy as np
import torch
import glob
import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def print_gpu_memory_info():
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)

        print(f"GPU {i}:")
        print(f"  Total memory: {total_memory / (1024 ** 3):.2f} GB")
        print(f"  Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
        print(f"  Reserved memory: {reserved_memory / (1024 ** 3):.2f} GB")
        print()

def main(hparams):


    model = network(hparams=hparams)
    os.makedirs(hparams.output_path, exist_ok=True)


    if not hparams.pretrained is None:
        model.load_pretrained(hparams.pretrained)


    resume_ckpt = None
    if hparams.resume:
        checkpoints = glob.glob(os.path.join(hparams.output_path,'*.ckpt'))
        checkpoints.sort(key=os.path.getmtime)
        if len(checkpoints)>0:
            resume_ckpt = checkpoints[-1]
            print('Resuming from checkpoint: {0}'.format(resume_ckpt))


    model.set_augmentations(hparams.augmentations)


    print('Saving 20 data samples for sanity checks...')
    model.train_dataloader().dataset.test(os.path.join(hparams.output_path, 'samples'), num_files=20)


    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.output_path,
        filename=hparams.pipeline+'-{epoch:03d}-{step}',
        save_top_k=1,
        monitor='step',
        mode='max',
        verbose=True,
        every_n_epochs=1
    )

    logger = TensorBoardLogger(
        save_dir=hparams.log_path,
        name='lightning_logs_'+hparams.pipeline.lower()
    )

    trainer = Trainer(
        logger=logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],

        devices=hparams.gpus,
        accelerator="gpu",
        strategy="ddp",
        min_epochs=hparams.epochs,

        max_epochs=hparams.epochs

    )


    trainer.fit(model, ckpt_path=resume_ckpt)


if __name__ == '__main__':


    parent_parser = ArgumentParser(add_help=False)


    parent_parser.add_argument(
        '--output_path',
        type=str,
        default=r'results/experiment1',
        help="Output path."
    )

    parent_parser.add_argument(
        '--log_path',
        type=str,
        default=r'logs/logs_experiment1',
        help="Log path."
    )

    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help="Gpus."
    )

    parent_parser.add_argument(
        '--no_resume',
        dest='resume',
        action='store_false',
        default=True,
        help="No resume."
    )

    parent_parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        nargs='+',
        help="Pretrained."
    )

    parent_parser.add_argument(
        '--augmentations',
        type=str,
        default=None,
        help="Augmentations."
    )

    parent_parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help="Epochs."
    )

    parent_parser.add_argument(
        '--pipeline',
        type=str,
        default='DiffusionModel3D',
        help="Pipeline."
    )

    parent_args = parent_parser.parse_known_args()[0]


    print_gpu_memory_info()


    if parent_args.pipeline.lower() == 'diffusionmodel3d':
        from models.DiffusionModel3D import DiffusionModel3D as network
    elif parent_args.pipeline.lower() == 'diffusionmodel2d':
        from models.DiffusionModel2D import DiffusionModel2D as network
    else:
        raise ValueError('Pipeline {0} unknown.'.format(parent_args.pipeline))


    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()


    main(hyperparams)
