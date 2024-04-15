import os
import sys
from collections import OrderedDict

from argparse import ArgumentParser
from enum import IntEnum
from io import BytesIO

import datasets
import numpy as np
import torch
from datasets import IterableDataset
from torchvision import transforms
from itertools import repeat
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
from aitrunk.diffusion.ldm.autoencoder import load_compvis_autoencoder
from aitrunk.diffusion.ldm.clip import CLIPTextEncoder
from aitrunk.diffusion.ldm.model import DiffusionLitModel
from aitrunk.diffusion.ldm.unet_attention import CrossAttention
import torch.multiprocessing as mp
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def get_tensor_mem_size(x):
    return x.element_size() * x.nelement()

class TextImageLatentDataDumper(object):
    class Message(object):
        class Signal(IntEnum):
            PUT = 0
            COMPLETE = 1

        def __init__(self, sender, signal=Signal.PUT, data=None):
            self.sender = sender
            self.signal = signal
            self.data = data

        def __repr__(self):
            return f'(sender: {self.sender}, signal: {self.signal.name}, data: {len(self.data) if self.data else None})'

    def __init__(self, autoencoder, text_encoder, transform=None):
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.transform = transform

    def dump_to_binary(self, source_key, fn_out, fp16=True, n_workers=1):
        source_ds = datasets.load_dataset(source_key, split='train')
        dss = np.array_split(source_ds, n_workers)
        mp.set_start_method('spawn', force=True)
        # Shared queue within sub processes
        q = mp.Manager().Queue()
        pool = mp.Pool(processes=(n_workers + 1))

        # Run sub process for the loop of dumping latent data
        result = pool.apply_async(self._proc_save_latent_data, args=(q, fn_out))
        pool.starmap(self._proc_put_latent_data, zip(dss, repeat(fp16, n_workers), repeat(q, n_workers)))

        q.put(self.Message(sender=mp.current_process().name, signal=self.Message.Signal.COMPLETE))
        pool.close()
        pool.join()

        chunk_size, total_size = result.get()
        print(f'Done dump_to_binary, chunk_size={chunk_size}, total_size={total_size}')

    def _proc_put_latent_data(self, ds, fp16, q=None):
        for i, row in enumerate(ds):
            img = self.transform(row['image'])
            z_img = self.autoencoder.encode(img.unsqueeze(0)).sample().squeeze(0)
            z_txt = self.text_encoder([row['prompt']]).squeeze(0)
            if fp16:
                z_img = z_img.to(torch.float16)
                z_txt = z_txt.to(torch.float16)
            msg = self.Message(sender=mp.current_process().name, signal=self.Message.Signal.PUT, data=(z_img, z_txt))
            q.put(msg)
            print(f'Putting {i}th message {msg} into queue')

    def _proc_save_latent_data(self, q, fn_out):
        print(f'Start _proc_save_latent_data in {mp.current_process().name}')
        chunk_size = None
        total_size = 0
        with open(fn_out, 'wb') as fp_out:
            while True:
                msg = q.get()
                print(f'Got message {msg} from queue')
                if msg is not None:
                    if msg.signal == self.Message.Signal.COMPLETE:
                        print(f'Got COMPLETE signal, exiting')
                        break
                    else:
                        torch.save(msg.data, fp_out)
                        total_size = fp_out.tell()
                        if chunk_size is None:
                            chunk_size = total_size

                        print(f'Wrote data chunk_size={chunk_size}, total_size={total_size} into {fn_out}')

        print(f'Done to _proc_save_latent_data in {mp.current_process().name}')
        return chunk_size, total_size


def dump_latent_data(args):
    print(f'Start dump_latent_data with args: {args}')

    # source_ds = datasets.load_dataset(args.source, split='train')
    autoencoder, ae_config = load_compvis_autoencoder(args.autoencoder)
    autoencoder.requires_grad_(False)
    text_encoder = CLIPTextEncoder(model_id=args.clip)
    text_encoder.requires_grad_(False)
    transform = transforms.Compose([
        transforms.Resize(args.img_size[1]),
        transforms.ToTensor()
    ])
    n_workers = args.n_workers if 'n_workers' in args else 1
    dumper = TextImageLatentDataDumper(autoencoder, text_encoder, transform)
    dumper.dump_to_binary(args.source, args.fn_out, fp16=args.fp16, n_workers=n_workers)


class BatchTransform:
    def __init__(self, img_size=256, autoencoder=None, text_encoder=None, latent_scaling_factor=None):
        self.transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.latent_scaling_factor = latent_scaling_factor

    # @torch.no_grad()
    def __call__(self, batch):
        z_input = self.autoencoder.encode(torch.stack([self.transform(img) for img in batch['image']])).sample()
        z_input = z_input * self.latent_scaling_factor
        z_cond = self.text_encoder(batch['text'])
        z_uncond = self.text_encoder.empty_encoded(z_cond.shape[0])
        return {'input': z_input, 'cond': z_cond, 'uncond': z_uncond}


class ZPairDataGenerator:
    def __init__(self, fn_source, chunk_size, latent_scaling_factor=None):
        self.fn_source = fn_source
        self.chunk_size = chunk_size
        self.latent_scaling_factor = latent_scaling_factor

    def __call__(self):
        with open(self.fn_source, 'rb') as f:
            while chunk := f.read(self.chunk_size):
                z_pair = torch.load(BytesIO(chunk))
                # print(f'Current yield: z_pair[0].shape: {z_pair[0].shape}, z_pair[1].shape: {z_pair[1].shape}')
                yield {'input': z_pair[0] * self.latent_scaling_factor, 'cond': z_pair[1]}


def run_train(args):
    print(f'Start train with args: {args}')
    config = get_model_config(args)

    # train_ds = datasets.load_dataset(args.data, split='train')
    # batch_transform = BatchTransform(img_size, autoencoder, text_encoder, latent_scaling_factor=config['latent_scaling_factor'])
    # train_ds.set_transform(batch_transform, columns=['image', 'text'])
    train_ds = IterableDataset.from_generator(generator=ZPairDataGenerator(fn_source=args.fn_source,
                                                                           chunk_size=args.chunk_size,
                                                                           latent_scaling_factor=args.latent_scaling_factor))
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size)

    model = DiffusionLitModel(config=config)
    ds_config = {
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 4,
            },
            "offload_param": {
                "device": 'cpu',
                "pin_memory": True,
                "buffer_count": 5,
                "buffer_size": 1e8,
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }

    dirpath, filename = os.path.split(args.filepath)
    callbacks = [ModelCheckpoint(dirpath=dirpath,
                                 filename=os.path.splitext(filename)[0],
                                 monitor='train.loss',
                                 mode='min',
                                 save_top_k=1)]

    trainer = pl.Trainer(accelerator=args.accelerator,
                         devices=args.devices,
                         # strategy=args.strategy,
                         strategy=DeepSpeedStrategy(config=ds_config),
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         precision=args.precision,
                         max_epochs=args.max_epochs,
                         callbacks=callbacks,
                         enable_checkpointing=True)
    #  FlashAttention backward for head dim > 64 requires A100 or H100 GPUs
    #  as the implementation needs a large amount of shared memory.
    CrossAttention.use_flash_attention = False
    torch.set_autocast_enabled(True)
    trainer.fit(model, train_dataloader)

    if ds_config['zero_optimization'].get('stage', 1) > 1:
        # Convert the zero checkpoint to a single checkpoint file
        while True:
            if os.path.exists(args.filepath):
                print(f'Converting zero checkpoint {args.filepath} to single checkpoint file {args.filepath}/model.ckpt')
                convert_zero_checkpoint_to_fp32_state_dict(args.filepath, f'{args.filepath}/model.ckpt')
                break


def get_model_config(args):
    config = OrderedDict({
        'uncond_scale': 7.5,
        'n_steps': 1000,
        'beta_start': 0.0008,
        'beta_end': 0.012,
        'eps_model': {
            'channels': 320,
            'attention_levels': [0, 1, 2],
            'n_res_blocks': 2,
            'channel_multipliers': [1, 2, 4, 4],
            'n_heads': 8,
            'tf_layers': 1,
        }
    })
    d_z = args.z_shape[0]
    config['eps_model']['in_channels'] = d_z
    config['eps_model']['out_channels'] = d_z
    config['eps_model']['d_cond'] = args.d_cond
    return config


def gen_samples(args):
    print(f'Start gen_samples with args: {args}')
    config = get_model_config(args)
    model = DiffusionLitModel.load_from_checkpoint(args.ckpt, config=config)
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    print(f'Model loaded: {model}')


    autoencoder, ae_config = load_compvis_autoencoder(args.autoencoder)
    autoencoder.to(device)
    autoencoder.requires_grad_(False)
    text_encoder = CLIPTextEncoder(model_id=args.clip)
    text_encoder.to(device)
    text_encoder.requires_grad_(False)

    sampler = model.create_sampler(method=args.sampler)
    cond = text_encoder(args.prompts).squeeze(0)
    bs = len(args.prompts)
    z_imgs = sampler.sample(x_shape=args.z_shape,
                            cond=cond,
                            # uncond_scale=7.5,
                            # uncond=text_encoder.empty_encoded(bs),
                            n_steps=args.n_steps)
    imgs = autoencoder.decode(z_imgs / args.latent_scaling_factor)
    imgs = imgs.detach().cpu().numpy()
    fig, axes = plt.subplots(nrows=bs, ncols=1, figsize=(50, 50))
    for i, img in enumerate(imgs):
        img = img.transpose(1, 2, 0)
        img = np.interp(img, (img.min(), img.max()), (0, 1))
        axes[i].imshow(img)
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser('aitrunk.diffusion.ldm')
    parser.add_argument('--log_level', type=str, default='DEBUG')
    subparsers = parser.add_subparsers()

    # Arguments for sub command 'dump_latent_data'
    sub_parser = subparsers.add_parser('dump_latent_data')
    sub_parser.set_defaults(func=dump_latent_data)
    sub_parser.add_argument('--source', type=str, default='wtcherr/LAION10K')
    sub_parser.add_argument('--autoencoder', type=str, default='kl-f8')
    sub_parser.add_argument('--clip', type=str, default='openai/clip-vit-large-patch14')
    sub_parser.add_argument('--img_size', type=tuple, default=(3, 256, 256))
    sub_parser.add_argument('--fn_out', type=str, default='output/laion256_latent.pt')
    sub_parser.add_argument('--fp16', type=bool, default=True)
    sub_parser.add_argument('--n_workers', type=int, default=12)


    # Arguments for sub command 'run_train'
    sub_parser = subparsers.add_parser('run_train')
    sub_parser.set_defaults(func=run_train)
    sub_parser.add_argument('--fn_source', type=str, default='output/laion256_latent.pt')
    sub_parser.add_argument('--chunk_size', type=int, default=127399)
    sub_parser.add_argument('--z_shape', type=tuple, default=(4, 32, 32))
    sub_parser.add_argument('--d_cond', type=int, default=768)
    sub_parser.add_argument('--latent_scaling_factor', type=float, default=0.18215)
    sub_parser.add_argument('--batch_size', type=int, default=16)
    sub_parser.add_argument('--accelerator', type=str, default='gpu')
    sub_parser.add_argument('--devices', type=int, default=2)
    # sub_parser.add_argument('--strategy', type=str, default='deepspeed_stage_3')
    sub_parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    sub_parser.add_argument('--precision', type=int, default=16)
    sub_parser.add_argument('--max_epochs', type=int, default=30)
    sub_parser.add_argument('--n_workers', type=int, default=1)
    sub_parser.add_argument('--filepath', type=str, default='output/best_ldm.ckpt')

    # Arguments for sub command 'gen_sample'
    prompts = ["Call for Applications AFMA 2020",
               "Only a strong woman can work for AT&T | AT&T Shirt",
               "Deadlines... T Shirt",
               "Girl holding blank board Stock Photo",
               "Rocket Space Dog Costume",
               "Girl holding blank board Stock Photo"]
    sub_parser = subparsers.add_parser('gen_samples')
    sub_parser.set_defaults(func=gen_samples)
    sub_parser.add_argument('--ckpt', type=str, default='output/best_ldm.ckpt/model.ckpt')
    sub_parser.add_argument('--sampler', type=str, default='ddpm')
    sub_parser.add_argument('--autoencoder', type=str, default='kl-f8')
    sub_parser.add_argument('--z_shape', type=tuple, default=(4, 32, 32))
    sub_parser.add_argument('--latent_scaling_factor', type=float, default=0.18215)
    sub_parser.add_argument('--clip', type=str, default='openai/clip-vit-large-patch14')
    sub_parser.add_argument('--prompts', type=int, default=prompts)
    sub_parser.add_argument('--d_cond', type=int, default=768)
    sub_parser.add_argument('--n_steps', type=int, default=1000)

    args = parser.parse_args()
    args.func(args)
