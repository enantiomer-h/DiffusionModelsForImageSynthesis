#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified.
# @author: eschweiler
"Apply a diffusion model to image patches."


import os
import numpy as np
import torch
import csv
from skimage import io
from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser
from torch.autograd import Variable

from dataloader.h5_dataloader import MeristemH5Tiler as Tiler
from ThirdParty.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from utils.utils import print_timestamp


from models.DiffusionModel3D import DiffusionModel3D


SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):


    if not isinstance(hparams.overlap, (tuple, list)):
        hparams.overlap = (hparams.overlap,) * len(hparams.patch_size)
    if not isinstance(hparams.crop, (tuple, list)):
        hparams.crop = (hparams.crop,) * len(hparams.patch_size)
    assert all([p-2*o-2*c>0 for p,o,c in zip(hparams.patch_size, hparams.overlap, hparams.crop)]), 'Invalid combination of patch size, overlap and crop size.'


    model = network(hparams=hparams)
    print(type(model))
    print(type(DiffusionModel3D))

    model = DiffusionModel3D.load_from_checkpoint(hparams.ckpt_path)
    model = model.cuda()


    device="cuda" if torch.cuda.is_available() else "cpu"

    if 'diffusionmodel' in hparams.pipeline.lower():
        DiffusionTrainer = GaussianDiffusionTrainer(hparams.num_timesteps, schedule=hparams.diffusion_schedule).to(device)
        DiffusionSampler = GaussianDiffusionSampler(model, hparams.num_timesteps, t_start=hparams.timesteps_start,\
                                                    t_save=hparams.timesteps_save, t_step=hparams.timesteps_step,\
                                                    schedule=hparams.diffusion_schedule,\
                                                    mean_type='epsilon', var_type='fixedlarge').to(device)
    else :
        raise NotImplementedError()

    hparams.out_channels *= DiffusionSampler.__len__()


    tiler = Tiler(hparams.test_list, no_mask=hparams.input_batch=='image', no_img=hparams.input_batch=='mask',\
                  boundary_handling='none', reduce_dim='2d' in hparams.pipeline.lower(), **vars(hparams))
    fading_map = tiler.get_fading_map()
    fading_map = np.repeat(fading_map[np.newaxis,...], hparams.out_channels, axis=0)


    os.makedirs(hparams.output_path, exist_ok=True)
    file_checklist = []


    if hparams.num_files is None or hparams.num_files < 0:
        hparams.num_files = len(tiler.data_list)
    else:
        hparams.num_files = np.minimum(len(tiler.data_list), hparams.num_files)

    with torch.no_grad():

        for image_idx in range(hparams.num_files):


            if os.path.isfile(os.path.join(hparams.output_path, 'tmp_file_checklist.csv')):
                file_checklist = []
                with open(os.path.join(hparams.output_path, 'tmp_file_checklist.csv'), 'r') as f:
                    reader = csv.reader(f, delimiter=';')
                    for row in reader:
                        if not len(row)==0:
                            file_checklist.append(row[0])


            if not any([f==tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1] for f in file_checklist]):

                print_timestamp('_'*20)
                print_timestamp('Processing file {0}', [tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1]])


                tiler.set_data_idx(image_idx)


                working_size = tuple(np.max(np.array(tiler.locations), axis=0) - np.min(np.array(tiler.locations), axis=0) + np.array(hparams.patch_size))


                predicted_img = np.full((hparams.out_channels,)+working_size, 0, dtype=np.float32)
                norm_map = np.full((hparams.out_channels,)+working_size, 0, dtype=np.float32)

                for patch_idx in range(tiler.__len__()):

                    print_timestamp('Generating patch {0}/{1}...',(patch_idx+1, tiler.__len__()))


                    sample = tiler.__getitem__(patch_idx)


                    if hparams.blur_sigma>0:
                        for ndim in range(sample[hparams.input_batch].shape[0]-1):
                            sample[hparams.input_batch][ndim,...] = gaussian_filter(sample[hparams.input_batch][ndim,...], hparams.blur_sigma, order=0)

                    data = Variable(torch.from_numpy(sample[hparams.input_batch][np.newaxis,...]).cuda())
                    data = data.float()


                    if hparams.conditioning_mode == 'training_match':


                        pred_patch = data[:,0:1,...].clone()
                        cond = data[:,1:,...].clone()

                        print_timestamp('Conditioning mode: training_match')
                        print_timestamp('  Channels to denoise: {0}', [pred_patch.shape[1]])
                        print_timestamp('  Conditioning channels: {0}', [cond.shape[1]])

                    elif hparams.conditioning_mode == 'legacy':


                        pred_patch = data[:,:-1,...].clone()
                        cond = data[:,-1:,...].clone()

                        print_timestamp('Conditioning mode: legacy (backward compatibility)')
                        pass
                        print_timestamp('  Channels to denoise: {0}', [pred_patch.shape[1]])
                        print_timestamp('  Conditioning channels: {0}', [cond.shape[1]])
                    else:
                        raise ValueError('Unknown conditioning_mode: {0}. Use "training_match" or "legacy"'.format(hparams.conditioning_mode))


                    pred_patch,_,_ = DiffusionTrainer(pred_patch, hparams.timesteps_start-1)


                    _,pred_patch = DiffusionSampler(pred_patch, cond)


                    pred_patch = [p.cpu().data.numpy() for p in pred_patch]
                    pred_patch = np.array(pred_patch)


                    pred_patch = pred_patch[:,0,...]
                    pred_patch = np.reshape(pred_patch, (pred_patch.shape[0]*pred_patch.shape[1],)+pred_patch.shape[2:])


                    pred_patch = np.clip(pred_patch, hparams.clip[0], hparams.clip[1])


                    slicing = tuple(map(slice, (0,)+tuple(tiler.patch_start+tiler.global_crop_before), (hparams.out_channels,)+tuple(tiler.patch_end+tiler.global_crop_before)))


                    predicted_img[slicing] = predicted_img[slicing]+pred_patch*fading_map
                    norm_map[slicing] = norm_map[slicing]+fading_map


                norm_map = np.clip(norm_map, 1e-5, np.inf)
                predicted_img = predicted_img / norm_map


                slicing = tuple(map(slice, (0,)+tuple(tiler.global_crop_before), (hparams.out_channels,)+tuple(np.array(predicted_img.shape[1:])+np.array(tiler.global_crop_after))))
                predicted_img = predicted_img[slicing]


                predicted_img = np.transpose(predicted_img, (1,2,3,0))
                predicted_img = predicted_img.astype(np.float32)
                if hparams.out_channels > 1:
                    for channel in range(hparams.out_channels):
                        io.imsave(os.path.join(hparams.output_path, 'pred_'+str(channel)+'_'+os.path.split(tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1])[-1][:-3]+'.tif'), predicted_img[...,channel])
                else:
                    io.imsave(os.path.join(hparams.output_path, 'pred_'+os.path.split(tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1])[-1][:-3]+'.tif'), predicted_img[...,0])


                file_checklist.append(tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1])
                with open(os.path.join(hparams.output_path, 'tmp_file_checklist.csv'), 'w') as f:
                    writer = csv.writer(f, delimiter=';')
                    for check_file in file_checklist:
                        writer.writerow([check_file])

            else:
                print_timestamp('_'*20)
                ([tiler.data_list[image_idx][0 if hparams.input_batch == 'image' else 1]],)


        os.remove(os.path.join(hparams.output_path, 'tmp_file_checklist.csv'))


if __name__ == '__main__':


    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--output_path',
        type=str,
        default=r'results/experiment1',
        help="Output path."
    )

    parent_parser.add_argument(
        '--ckpt_path',
        type=str,
        default=r'results/experiment1/checkpoint.ckpt',
        help="Ckpt path."
    )

    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help="Gpus."
    )

    parent_parser.add_argument(
        '--overlap',
        type=int,
        default=(0,0,0),
        help="Overlap.",
        nargs='+'
    )

    parent_parser.add_argument(
        '--crop',
        type=int,
        default=(0,0,0),
        help="Crop.",
        nargs='+'
    )

    parent_parser.add_argument(
        '--input_batch',
        type=str,
        default='mask',
        help="Input batch."
    )

    parent_parser.add_argument(
        '--conditioning_mode',
        type=str,
        default='training_match',
        choices=['training_match', 'legacy'],
        help="Conditioning mode."
    )

    parent_parser.add_argument(
        '--clip',
        type=float,
        default=(-1000.0, 1000.0),
        help="Clip.",
        nargs='+'
    )

    parent_parser.add_argument(
        '--num_files',
        type=int,
        default=1,
        help="Num files."
    )

    parent_parser.add_argument(
        '--timesteps_start',
        type=int,
        default=400,
        help="Timesteps start."
    )

    parent_parser.add_argument(
        '--timesteps_save',
        type=int,
        default=100,
        help="Timesteps save."
    )

    parent_parser.add_argument(
        '--blur_sigma',
        type=int,
        default=1,
        help="Blur sigma."
    )

    parent_parser.add_argument(
        '--timesteps_step',
        type=int,
        default=1,
        help="Timesteps step."
    )

    parent_parser.add_argument(
        '--pipeline',
        type=str,
        default='DiffusionModel3D',
        help="Pipeline."
    )

    parent_args = parent_parser.parse_known_args()[0]


    if parent_args.pipeline.lower() == 'diffusionmodel3d':
        from models.DiffusionModel3D import DiffusionModel3D as network
    elif parent_args.pipeline.lower() == 'diffusionmodel2d':
        from models.DiffusionModel2D import DiffusionModel2D as network
    else:
        raise ValueError('Unknown pipeline "{0}".'.format(parent_args.pipeline))


    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()


    main(hyperparams)
