# -*- coding: utf-8 -*-
# Modified.
# @author: Nutzer


import os
import json
import torch
import numpy as np


def create_defaults(save_dir):

    augmentation_dict = {}
    augmentation_dict['prob'] = 0.5
    augmentation_dict['queue'] = ['scale', 'noise', 'shuffle', 'inpaint', 'decline', 'variance_sampling', 'stretch', 'shrink', 'anisotropy']

    augmentation_dict['scale_min'] = 0.6
    augmentation_dict['scale_max'] = 1.2

    augmentation_dict['noise_mean'] = 0
    augmentation_dict['noise_std'] = 0.2

    augmentation_dict['shuffle_size'] = (25,25,25)
    augmentation_dict['shuffle_count'] = 3

    augmentation_dict['inpaint_size'] = (15,15,15)
    augmentation_dict['inpaint_count'] = 3

    augmentation_dict['decline_axes'] = [0,1,2]

    augmentation_dict['sampling_mean_weight'] = 1
    augmentation_dict['sampling_std_weight'] = 1

    augmentation_dict['stretch_factor'] = 2
    augmentation_dict['stretch_axes'] = [0,1,2]

    augmentation_dict['shrink_factor'] = 2
    augmentation_dict['shrink_axes'] = [0,1,2]

    augmentation_dict['anisotropy_factor'] = 2
    augmentation_dict['anisotropy_axes'] = [0,1,2]

    augmentation_dict['permute_dim'] = False

    with open(os.path.join(save_dir, 'augments_default.json'), 'w') as file_handle:
        json.dump(augmentation_dict, file_handle)


class intensity_augmenter():

    def __init__(self, augmentation_dict={}):

        self.augmentation_dict = augmentation_dict
        self.dict_sanitycheck()


    def dict_sanitycheck(self):


        _=self.augmentation_dict.setdefault('prob', 0)
        _=self.augmentation_dict.setdefault('queue', [])


        _=self.augmentation_dict.setdefault('scale_min', 0.6)
        _=self.augmentation_dict.setdefault('scale_max', 1.2)


        _=self.augmentation_dict.setdefault('noise_mean', 0)
        _=self.augmentation_dict.setdefault('noise_std', 0.1)


        _=self.augmentation_dict.setdefault('shuffle_size', (25,25,25))
        _=self.augmentation_dict.setdefault('shuffle_count', 3)
        assert len(self.augmentation_dict['shuffle_size'])==3, 'Shuffle window size must be 3-dimensional.'


        _=self.augmentation_dict.setdefault('inpaint_size', (15,15,15))
        _=self.augmentation_dict.setdefault('inpaint_count', 3)
        assert len(self.augmentation_dict['inpaint_size'])==3, 'Inpainting window size must be 3-dimensional.'


        _=self.augmentation_dict.setdefault('decline_axes', [0,1,2])
        if not isinstance(self.augmentation_dict['decline_axes'], (list,tuple)):
            self.augmentation_dict['decline_axes'] = list(self.augmentation_dict['decline_axes'])


    def linear_scaling(self, patch, min_val=1, max_val=1):
        patch_min, patch_max = np.min(patch), np.max(patch)
        patch = patch * np.random.uniform(low=min_val, high=max_val)
        patch = np.clip(patch, patch_min, patch_max)
        return patch


    def additive_noise(self, patch, mean=0, std=0):
        patch_min, patch_max = np.min(patch), np.max(patch)
        patch = patch + np.random.normal(loc=mean, scale=std, size=patch.shape)
        patch = np.clip(patch, patch_min, patch_max)
        return patch


    def windowed_shuffle(self, patch, shuffle_size=(5,5,5), shuffle_count=0):

        for num_window in range(shuffle_count):


            window_start = [np.random.randint(0, np.maximum(1,patch_dim-window_dim)) for window_dim,patch_dim in zip(shuffle_size, patch.shape)]
            window_end = [start+window_dim for start,window_dim in zip(window_start, shuffle_size)]
            slicing = tuple(map(slice, window_start, window_end))


            window = patch[slicing]
            np.random.shuffle(window)


            patch[slicing] = window

        return patch


    def inpaint(self, patch, inpaint_size=(5,5,5), inpaint_count=0):
        assert len(inpaint_size)==3, 'Window size must be 3-dimensional.'
        for num_window in range(inpaint_count):


            window_start = [np.random.randint(0, np.maximum(1,patch_dim-window_dim)) for window_dim,patch_dim in zip(inpaint_size, patch.shape)]
            window_end = [start+window_dim for start,window_dim in zip(window_start, inpaint_size)]
            slicing = tuple(map(slice, window_start, window_end))


            patch[slicing] = np.random.uniform(low=patch.min(), high=patch.max())

        return patch


    def intensity_decline(self, patch, decline_axes=[0]):


        decline_axis = np.random.choice(decline_axes)


        decline_extend = np.random.uniform(0,1)
        decline = np.linspace(0, 1, num=int(decline_extend*patch.shape[decline_axis]))
        decline = np.pad(decline, (0,patch.shape[decline_axis]-len(decline)), constant_values=1)
        decline = np.expand_dims(decline, axis=tuple([i for i in range(patch.ndim) if not i==decline_axis]))


        patch = patch * decline

        return patch


    def apply(self, patch):

        if np.any(np.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the scale augmentation.')
        if np.random.rand() <= self.augmentation_dict['prob'] and 'scale' in self.augmentation_dict['queue']:
            patch = self.linear_scaling(patch, min_val=self.augmentation_dict['scale_min'], max_val=self.augmentation_dict['scale_max'])

        if np.any(np.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the noise augmentation.')
        if np.random.rand() <= self.augmentation_dict['prob'] and 'noise' in self.augmentation_dict['queue']:
            patch = self.additive_noise(patch, mean=self.augmentation_dict['noise_mean'], std=self.augmentation_dict['noise_std'])

        if np.any(np.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the shuffle augmentation.')
        if np.random.rand() <= self.augmentation_dict['prob'] and 'shuffle' in self.augmentation_dict['queue']:
            patch = self.windowed_shuffle(patch, shuffle_size=self.augmentation_dict['shuffle_size'], shuffle_count=self.augmentation_dict['shuffle_count'])

        if np.any(np.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the inpaint augmentation.')
        if np.random.rand() <= self.augmentation_dict['prob'] and 'inpaint' in self.augmentation_dict['queue']:
            patch = self.inpaint(patch, inpaint_size=self.augmentation_dict['inpaint_size'], inpaint_count=self.augmentation_dict['inpaint_count'])

        if np.any(np.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the decline augmentation.')
        if np.random.rand() <= self.augmentation_dict['prob'] and 'decline' in self.augmentation_dict['queue']:
            patch = self.intensity_decline(patch, decline_axes=self.augmentation_dict['decline_axes'])

        return patch


class intensity_augmenter_pytorch():

    def __init__(self, augmentation_dict={}):

        self.augmentation_dict = augmentation_dict
        self.dict_sanitycheck()


    def dict_sanitycheck(self):


        _=self.augmentation_dict.setdefault('prob', 0)
        _=self.augmentation_dict.setdefault('queue', [])


        _=self.augmentation_dict.setdefault('scale_min', 0.6)
        _=self.augmentation_dict.setdefault('scale_max', 1.2)


        _=self.augmentation_dict.setdefault('noise_mean', 0)
        _=self.augmentation_dict.setdefault('noise_std', 0.2)


        _=self.augmentation_dict.setdefault('shuffle_size', (25,25,25))
        _=self.augmentation_dict.setdefault('shuffle_count', 3)
        assert len(self.augmentation_dict['shuffle_size'])==3, 'Shuffle window size must be 3-dimensional.'


        _=self.augmentation_dict.setdefault('decline_axes', [0,1,2])
        if not isinstance(self.augmentation_dict['decline_axes'], (list,tuple)):
            self.augmentation_dict['decline_axes'] = list(self.augmentation_dict['decline_axes'])


    def linear_scaling(self, patch, min_val=1, max_val=1):
        patch = patch * (torch.tensor(min_val, dtype=patch.dtype, device=patch.device) + max_val*torch.rand(1, dtype=patch.dtype, device=patch.device))
        return patch


    def additive_noise(self, patch, mean=0, std=0):
        patch = patch + torch.randn_like(patch)*(std**0.5) + torch.tensor(mean, dtype=patch.dtype, device=patch.device)
        return patch


    def windowed_scaling(self, patch, shuffle_size=(5,5,5), shuffle_count=0):

        for num_window in range(shuffle_count):


            window_start = [0,]*(patch.ndim-3) + [np.random.randint(0, np.maximum(1,patch_dim-window_dim)) for window_dim,patch_dim in zip(shuffle_size, patch.shape[patch.ndim-3:])]
            window_end = list(patch.shape[0:patch.ndim-3]) + [start+window_dim for start,window_dim in zip(window_start[patch.ndim-3:], shuffle_size)]
            slicing = tuple(map(slice, window_start, window_end))


            window = torch.ones_like(patch)
            window[slicing] = torch.rand(tuple(np.array(window_end)-np.array(window_start)), dtype=patch.dtype, device=patch.device)
            patch = patch * window

        return patch


    def intensity_decline(self, patch, decline_axes=[0]):


        decline_axis = decline_axes[torch.randint(0,len(decline_axes),(1,))]


        decline_extend = torch.rand(1, dtype=patch.dtype, device=patch.device)
        decline = torch.linspace(0, 1, steps=int(decline_extend*patch.shape[decline_axis]), dtype=patch.dtype, device=patch.device)
        decline = torch.nn.functional.pad(decline, (0,patch.shape[decline_axis]-len(decline)), mode='constant', value=1)
        for expand_dim in [i for i in range(patch.ndim) if not i==decline_axis]:
            decline = torch.unsqueeze(decline, expand_dim)


        patch = patch * decline

        return patch


    def apply(self, patch):

        if torch.any(torch.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the scale augmentation.')
        if torch.rand(1) <= self.augmentation_dict['prob'] and 'scale' in self.augmentation_dict['queue']:
            patch = self.linear_scaling(patch, min_val=self.augmentation_dict['scale_min'], max_val=self.augmentation_dict['scale_max'])

        if torch.any(torch.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the noise augmentation.')
        if torch.rand(1) <= self.augmentation_dict['prob'] and 'noise' in self.augmentation_dict['queue']:
            patch = self.additive_noise(patch, mean=self.augmentation_dict['noise_mean'], std=self.augmentation_dict['noise_std'])

        if torch.any(torch.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the shuffle augmentation.')
        if torch.rand(1) <= self.augmentation_dict['prob'] and 'shuffle' in self.augmentation_dict['queue']:
            patch = self.windowed_scaling(patch, shuffle_size=self.augmentation_dict['shuffle_size'], shuffle_count=self.augmentation_dict['shuffle_count'])

        if torch.any(torch.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the decline augmentation.')
        if torch.rand(1) <= self.augmentation_dict['prob'] and 'decline' in self.augmentation_dict['queue']:
            patch = self.intensity_decline(patch, decline_axes=self.augmentation_dict['decline_axes'])


        return patch


class geometry_augmenter():

    def __init__(self, augmentation_dict={}):

        self.augmentation_dict = augmentation_dict
        self.dict_sanitycheck()


        self.prob_anisotropy = np.random.rand()
        self.prob_stretching = np.random.rand()


    def dict_sanitycheck(self):


        _=self.augmentation_dict.setdefault('prob', 0)
        _=self.augmentation_dict.setdefault('queue', [])


        _=self.augmentation_dict.setdefault('stretch_factor', 2)
        _=self.augmentation_dict.setdefault('stretch_axes', [0,1,2])
        if not isinstance(self.augmentation_dict['stretch_axes'], (list,tuple)):
            self.augmentation_dict['stretch_axes'] = list(self.augmentation_dict['stretch_axes'])
        _=self.augmentation_dict.setdefault('stretch_axis', self.augmentation_dict['stretch_axes'][0])


        _=self.augmentation_dict.setdefault('shrink_factor', 2)
        _=self.augmentation_dict.setdefault('shrink_axes', [0,1,2])
        if not isinstance(self.augmentation_dict['shrink_axes'], (list,tuple)):
            self.augmentation_dict['shrink_axes'] = list(self.augmentation_dict['shrink_axes'])
        _=self.augmentation_dict.setdefault('shrink_axis', self.augmentation_dict['shrink_axes'][0])


        _=self.augmentation_dict.setdefault('anisotropy_factor', 2)
        _=self.augmentation_dict.setdefault('anisotropy_axes', [0,1,2])
        if self.augmentation_dict['anisotropy_factor']%1 > 0:
            print('The anisotropy factor needs to be integer valued. Setting "anisotropy_factor" to {0}.'.format(int(self.augmentation_dict['anisotropy_factor'])))
        self.augmentation_dict['anisotropy_factor'] = int(self.augmentation_dict['anisotropy_factor'])
        if not isinstance(self.augmentation_dict['anisotropy_axes'], (list,tuple)):
            self.augmentation_dict['anisotropy_axes'] = list(self.augmentation_dict['anisotropy_axes'])
        _=self.augmentation_dict.setdefault('anisotropy_axis', self.augmentation_dict['anisotropy_axes'][0])


        _=self.augmentation_dict.setdefault('permute_dim', False)


    def stretching(self, patch, stretch_factor=1, stretch_axis=0):


        orig_shape = patch.shape


        patch_crop = [int(np.ceil(p/stretch_factor)) if stretch_axis==i else p for i,p in enumerate(patch.shape)]
        patch = patch[:patch_crop[0],\
                      :patch_crop[1],\
                      :patch_crop[2]]
        patch = np.repeat(patch, stretch_factor, axis=stretch_axis)
        patch = patch[:orig_shape[0], :orig_shape[1], :orig_shape[2]]


        assert patch.shape == orig_shape, 'Shape missmatch after stretching. {0} to {1}.'.format(orig_shape, patch.shape)

        return patch


    def shrinking(self, patch, shrink_factor=1, shrink_axis=0):


        orig_shape = patch.shape


        if shrink_axis==0:
            patch = patch[::shrink_factor,...]
        elif shrink_axis==1:
            patch = patch[:,::shrink_factor,...]
        elif shrink_axis==2:
            patch = patch[:,:,::shrink_factor,...]
        else:
            raise NotImplementedError('Anisotropy transform not implemented for axis "{0}".'.format(shrink_axis))


        padding = [(0,o-p) for o,p in zip(orig_shape, patch.shape)]
        patch = np.pad(patch, padding, mode='reflect')


        assert patch.shape == orig_shape, 'Shape missmatch after shrinking transform. {0} to {1}.'.format(orig_shape, patch.shape)

        return patch


    def anisotropy(self, patch, anisotropy_factor=1, anisotropy_axis=0):


        orig_shape = patch.shape


        if anisotropy_axis==0:
            patch = patch[::anisotropy_factor,...]
        elif anisotropy_axis==1:
            patch = patch[:,::anisotropy_factor,...]
        elif anisotropy_axis==2:
            patch = patch[:,:,::anisotropy_factor,...]
        else:
            raise NotImplementedError('Anisotropy transform not implemented for axis "{0}".'.format(anisotropy_axis))


        patch = np.repeat(patch, anisotropy_factor, axis=anisotropy_axis)


        assert patch.shape == orig_shape, 'Shape missmatch after anisotropy transform. {0} to {1}.'.format(orig_shape, patch.shape)

        return patch


    def apply(self, patch, reset=True):


        if np.any(np.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the anisotropy augmentation.')
        if reset:
            self.prob_anisotropy = np.random.rand()
            self.augmentation_dict['anisotropy_axis'] = np.random.choice(self.augmentation_dict['anisotropy_axes'])
        if self.prob_anisotropy <= self.augmentation_dict['prob'] and 'anisotropy' in self.augmentation_dict['queue']:
            patch = self.anisotropy(patch, anisotropy_factor=self.augmentation_dict['anisotropy_factor'], anisotropy_axis=self.augmentation_dict['anisotropy_axis'])

        if np.any(np.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the stretching augmentation.')
        if reset:
            self.prob_stretching = np.random.rand()
            self.augmentation_dict['stretch_axis'] = np.random.choice(self.augmentation_dict['stretch_axes'])
        if self.prob_stretching <= self.augmentation_dict['prob'] and 'stretch' in self.augmentation_dict['queue']:
            patch = self.stretching(patch, stretch_factor=self.augmentation_dict['stretch_factor'], stretch_axis=self.augmentation_dict['stretch_axis'])

        if np.any(np.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the shrinking augmentation.')
        if reset:
            self.prob_shrinking = np.random.rand()
            self.augmentation_dict['shrink_axis'] = np.random.choice(self.augmentation_dict['shrink_axes'])
        if self.prob_shrinking <= self.augmentation_dict['prob'] and 'shrink' in self.augmentation_dict['queue']:
            patch = self.shrinking(patch, shrink_factor=self.augmentation_dict['shrink_factor'], shrink_axis=self.augmentation_dict['shrink_axis'])


        return patch


class geometry_augmenter_pytorch():

    def __init__(self, augmentation_dict={}):

        self.augmentation_dict = augmentation_dict
        self.dict_sanitycheck()


        self.prob_anisotropy = np.random.rand()
        self.prob_stretching = np.random.rand()


    def dict_sanitycheck(self):


        _=self.augmentation_dict.setdefault('prob', 0)
        _=self.augmentation_dict.setdefault('queue', [])


        _=self.augmentation_dict.setdefault('stretch_factor', 2)
        _=self.augmentation_dict.setdefault('stretch_axes', [0,1,2])
        if not isinstance(self.augmentation_dict['stretch_axes'], (list,tuple)):
            self.augmentation_dict['stretch_axes'] = list(self.augmentation_dict['stretch_axes'])
        _=self.augmentation_dict.setdefault('stretch_axis', self.augmentation_dict['stretch_axes'][0])


        _=self.augmentation_dict.setdefault('shrink_factor', 2)
        _=self.augmentation_dict.setdefault('shrink_axes', [0,1,2])
        if not isinstance(self.augmentation_dict['shrink_axes'], (list,tuple)):
            self.augmentation_dict['shrink_axes'] = list(self.augmentation_dict['shrink_axes'])
        _=self.augmentation_dict.setdefault('shrink_axis', self.augmentation_dict['shrink_axes'][0])


        _=self.augmentation_dict.setdefault('anisotropy_factor', 2)
        _=self.augmentation_dict.setdefault('anisotropy_axes', [0,1,2])
        if self.augmentation_dict['anisotropy_factor']%1 > 0:
            print('The anisotropy factor needs to be integer valued. Setting "anisotropy_factor" to {0}.'.format(int(self.augmentation_dict['anisotropy_factor'])))
        self.augmentation_dict['anisotropy_factor'] = int(self.augmentation_dict['anisotropy_factor'])
        if not isinstance(self.augmentation_dict['anisotropy_axes'], (list,tuple)):
            self.augmentation_dict['anisotropy_axes'] = list(self.augmentation_dict['anisotropy_axes'])
        _=self.augmentation_dict.setdefault('anisotropy_axis', self.augmentation_dict['anisotropy_axes'][0])


        _=self.augmentation_dict.setdefault('permute_dim', False)


    def stretching(self, patch, stretch_factor=1, stretch_axis=0):


        orig_shape = patch.size()


        patch_crop = [int(np.ceil(p/stretch_factor)) if stretch_axis==i else p for i,p in enumerate(patch.size())]
        patch = patch[:,:,\
                      :patch_crop[0],\
                      :patch_crop[1],\
                      :patch_crop[2]]
        stretch_vector = [1,]*patch.ndim
        stretch_vector[stretch_axis+2]=stretch_factor
        patch = patch.repeat(*stretch_vector)
        patch = patch[:,:, :orig_shape[2], :orig_shape[3], :orig_shape[4]]


        assert patch.size() == orig_shape, 'Shape missmatch after stretching. {0} to {1}.'.format(orig_shape, patch.size())

        return patch


    pass


    def anisotropy(self, patch, anisotropy_factor=1, anisotropy_axis=0):


        orig_shape = patch.size()


        if anisotropy_axis==0:
            patch = patch[:,:,::anisotropy_factor,...]
        elif anisotropy_axis==1:
            patch = patch[:,:,:,::anisotropy_factor,...]
        elif anisotropy_axis==2:
            patch = patch[:,:,:,:,::anisotropy_factor,...]
        else:
            raise NotImplementedError('Anisotropy transform not implemented for axis "{0}".'.format(anisotropy_axis))


        anisotropy_vector = [1,]*patch.ndim
        anisotropy_vector[anisotropy_axis+2]=anisotropy_factor
        patch = patch.repeat(*anisotropy_vector)


        assert patch.size() == orig_shape, 'Shape missmatch after anisotropy transform. {0} to {1}.'.format(orig_shape, patch.size())

        return patch


    def apply(self, patch, reset=True):


        if torch.any(torch.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the anisotropy augmentation.')
        if reset:
            self.prob_anisotropy = torch.rand(1)
            self.augmentation_dict['anisotropy_axis'] = np.random.choice(self.augmentation_dict['anisotropy_axes'])
        if self.prob_anisotropy <= self.augmentation_dict['prob'] and 'anisotropy' in self.augmentation_dict['queue']:
            patch = self.anisotropy(patch, anisotropy_factor=self.augmentation_dict['anisotropy_factor'], anisotropy_axis=self.augmentation_dict['anisotropy_axis'])

        if torch.any(torch.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the stretching augmentation.')
        if reset:
            self.prob_stretching = torch.rand(1)
            self.augmentation_dict['stretch_axis'] = np.random.choice(self.augmentation_dict['stretch_axes'])
        if self.prob_stretching <= self.augmentation_dict['prob'] and 'stretch' in self.augmentation_dict['queue']:
            patch = self.stretching(patch, stretch_factor=self.augmentation_dict['stretch_factor'], stretch_axis=self.augmentation_dict['stretch_axis'])

        if torch.any(torch.isnan(patch)): raise ValueError('Encountered "NaN" value before applying the shrinking augmentation.')
        if reset:
            self.prob_shrinking = torch.rand(1)
            self.augmentation_dict['shrink_axis'] = np.random.choice(self.augmentation_dict['shrink_axes'])
        if self.prob_shrinking <= self.augmentation_dict['prob'] and 'shrink' in self.augmentation_dict['queue']:
            patch = self.shrinking(patch, shrink_factor=self.augmentation_dict['shrink_factor'], shrink_axis=self.augmentation_dict['shrink_axis'])

        return patch


def variance_sampling(patch, var, mean_weight=1, var_weight=1):
    assert patch.shape==var.shape, 'Patch size and variance size must be the same.'

    patch = np.random.normal(mean_weight*patch, var_weight*var)
    patch = np.clip(patch, 0, 1)

    return patch


def variance_sampling_pytorch(patch, var, mean_weight=1, var_weight=1):
    assert patch.shape==var.shape, 'Patch size and variance size must be the same.'

    patch = torch.normal(mean_weight*patch, var_weight*var)
    patch = torch.clamp(patch, 0, 1)

    return patch


def reparametrize(patch, var, mean_weight=1, var_weight=1):
    assert patch.shape==var.shape, 'Patch size and variance size must be the same.'

    patch = mean_weight*patch + var_weight*np.random.randn(*var.shape)*np.exp(var/2)

    return patch


def reparametrize_pytorch(patch, var, mean_weight=1, var_weight=1):
    assert patch.shape==var.shape, 'Patch size and variance size must be the same.'

    patch = mean_weight*patch + var_weight*torch.randn_like(var)*torch.exp(var/2)

    return patch


def diffusion_step(x_0, t, beta):

    pass
