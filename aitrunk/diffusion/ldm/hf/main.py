from collections import OrderedDict

from argparse import ArgumentParser

import datasets
import torch
from torchvision import transforms
from transformers import TrainingArguments

from aitrunk.diffusion.ldm.trainer import LDMTrainer
from aitrunk.diffusion.ldm.autoencoder import load_compvis_autoencoder
from aitrunk.diffusion.ldm.clip import CLIPTextEncoder
from aitrunk.diffusion.ldm.hf.ldm import LDM
from aitrunk.diffusion.ldm.unet_attention import CrossAttention


def get_tensor_mem_size(x):
    return x.element_size() * x.nelement()


def save_latent_data(args):
    print(f'Start save_latent_data with args: {args}')
    source_ds = datasets.load_dataset(args.source, split='train', streaming=True)
    autoencoder, ae_config = load_compvis_autoencoder(args.autoencoder)
    text_encoder = CLIPTextEncoder(model_id=args.clip)
    transform = transforms.Compose([
        transforms.Resize(args.img_size[1]),
        transforms.ToTensor()
    ])
    with open(args.fn_out, 'wb') as f:
        for row in source_ds:
            img = transform(row['image'])
            z_img = autoencoder.encode(img.unsqueeze(0)).sample()
            z_txt = text_encoder([row['text']])
            print(f'Saving z_img: {z_img.shape}, {get_tensor_mem_size(z_img) / 1000}K, z_txt: {z_txt.shape}, '
                  f'{get_tensor_mem_size(z_txt) / 1000}K')
            torch.save((z_img, z_txt), f)


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


def run_train(args):
    print(f'Start train with args: {args}')
    config = OrderedDict({
        'uncond_scale': 7.5,
        'latent_scaling_factor': 0.18215,
        'n_steps': 1000,
        'beta_start': 0.0008,
        'beta_end': 0.012,
        'autoencoder': 'kl-f8',
        'cond_encoder': 'openai/clip-vit-large-patch14',
        'eps_model': {
            'channels': 320,
            'attention_levels': [0, 1, 2],
            'n_res_blocks': 2,
            'channel_multipliers': [1, 2, 4, 4],
            'n_heads': 8,
            'tf_layers': 1,
        }
    })
    img_size = 256
    autoencoder, ae_config = load_compvis_autoencoder(config['autoencoder'])
    autoencoder.requires_grad_(False)
    text_encoder = CLIPTextEncoder(model_id=config['cond_encoder'])
    text_encoder.requires_grad_(False)
    d_z = ae_config['model']['params']['embed_dim']
    d_cond = text_encoder.hidden_size
    config['eps_model']['in_channels'] = d_z
    config['eps_model']['out_channels'] = d_z
    config['eps_model']['d_cond'] = d_cond

    train_ds = datasets.load_dataset(args.data, split='train')
    batch_transform = BatchTransform(img_size, autoencoder, text_encoder, latent_scaling_factor=config['latent_scaling_factor'])
    train_ds.set_transform(batch_transform, columns=['image', 'text'])
    # train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    CrossAttention.use_flash_attention = False
    torch.set_autocast_enabled(True)

    model = LDM(config=config)
    train_args = TrainingArguments(per_device_train_batch_size=2,
                                   num_train_epochs=1,
                                   do_train=True,
                                   output_dir='../output',
                                   remove_unused_columns=False)
    trainer = LDMTrainer(model=model,
                         args=train_args,
                         train_dataset=train_ds)
    trainer.train()

if __name__ == '__main__':
    parser = ArgumentParser('aitrunk.diffusion.ldm')
    parser.add_argument('--log_level', type=str, default='DEBUG')
    subparsers = parser.add_subparsers()

    # Arguments for sub command 'generate_train_data'
    sub_parser = subparsers.add_parser('gen_train_data')
    sub_parser.set_defaults(func=save_latent_data)
    sub_parser.add_argument('--autoencoder', type=str, default='kl-f8')
    sub_parser.add_argument('--clip', type=str, default='openai/clip-vit-large-patch14')
    sub_parser.add_argument('--source', type=str, default='che111/laion256')
    sub_parser.add_argument('--img_size', type=tuple, default=(3, 256, 256))
    sub_parser.add_argument('--outdir', type=str, default='output/sample_laion256')

    sub_parser = subparsers.add_parser('run_train')
    sub_parser.set_defaults(func=run_train)
    sub_parser.add_argument('--data', type=str, default='che111/laion256')
    sub_parser.add_argument('--batch_size', type=int, default=8)
    sub_parser.add_argument('--accelerator', type=str, default='gpu')
    sub_parser.add_argument('--devices', type=int, default=2)
    # sub_parser.add_argument('--strategy', type=str, default='deepspeed_stage_3')
    sub_parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    sub_parser.add_argument('--precision', type=int, default=16)
    sub_parser.add_argument('--max_epochs', type=int, default=1)
    sub_parser.add_argument('--n_workers', type=int, default=1)
    args = parser.parse_args()
    args.func(args)