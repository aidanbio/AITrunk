import copy
import os

from argparse import ArgumentParser
from enum import IntEnum
from io import BytesIO
import logging
import pandas as pd
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from peft import LoraConfig, get_peft_model, AutoPeftModel
# import bitsandbytes as bnb
from transformers import AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig
from tcrvae.data import EpitopeComplexDataset, TokenizingCollator
from tcrvae.model import LangVAELitModel
from tcrvae.common import TorchUtils

## Logger
logging.config.fileConfig('../config/logging.conf')
logger = logging.getLogger('tcrvae')


def gen_train_data(args):
    logger.info(f'Start gen_train_data with args: {args}')
    EpitopeComplexDataset.FN_DATA_CONFIG = args.data_config
    for data_key in args.data.split(','):
        logger.info(f'Generating data for {data_key}')
        try:
            ds = EpitopeComplexDataset.from_key(data_key, args=args)
            logger.info(f'Done to generate data for {data_key}, the number of data: {len(ds)}')
        except Exception as e:
            logger.error(f'Failed to generate data for {data_key}: {e}')


def create_peft_model(args):
    def create_bnb_config(bits=4):
        if bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        elif bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None

    def create_peft_config(modules):
        config = LoraConfig(
            r=16,  # dimension of the updated matrices
            lora_alpha=32,  # parameter for scaling
            target_modules=modules,
            lora_dropout=0.05,  # dropout probability for layers
            bias="none",
            # task_type=TaskType.TOKEN_CLS
        )
        return config

    def find_all_linear_names(model, bits=None):
        cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def print_trainable_parameters(model, use_4bit=False):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(f"all params: {int(all_param):,d} || trainable params: {int(trainable_params):,d} "
              f"|| trainable %: {100 * trainable_params / all_param}")

    logger.info(f'Start create_peft_model with args: {args}')
    model = AutoModelForMaskedLM.from_pretrained(args.target_model,
                                                 # quantization_config=create_bnb_config(bits=args.bits),
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    # Using the prepare_model_for_kbit_training method from PEFT
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=model.supports_gradient_checkpointing)

    # Create peft model with lora module names
    modules = find_all_linear_names(model)
    peft_config = create_peft_config(modules)
    logger.info(f'Creating peft model with lora config: {peft_config}')
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # print_trainable_parameters(model, use_4bit=(args.bits == 4))

    # Save the model
    logger.info(f'Saving the model and tokenizer to {args.outdir}')
    model.save_pretrained(args.outdir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    tokenizer.save_pretrained(args.outdir)
    logger.info(f'Done to create_peft_model.')

    # Merge the model with LORA adaptor
    # model = AutoPeftModel.from_pretrained(args.outdir, device_map="auto", torch_dtype=torch.bfloat16)
    # model = model.merge_and_unload()
    # model.save_pretrained(args.outdir, safe_serialization=True)


# def get_last_ckpt_path(outdir, filename):
#     fn_part = f"{outdir}/{filename}"
#     fns = glob.glob(f'{fn_part}*.ckpt')
#     if len(fns) == 0:
#         logger.warning(f'No checkpoint files found with pattern: {fn_part}*.ckpt')
#         return f'{fn_part}.ckpt'
#
#     logger.debug(f'ckpt files: {fns}')
#     the = np.argmax([StrUtils.search_digit(fn[len(fn_part):], default=0, last=True) for fn in fns])
#     return fns[the]


def run_train(args):
    print(f'Start train with args: {args}')
    # Load the pre-trained protein language model and tokenizer
    plm = AutoPeftModel.from_pretrained(args.plm_name_or_path, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.plm_name_or_path)
    logger.info(f'Loaded the pre-trained protein language model and tokenizer from {args.plm_name_or_path}: {plm}')
    EpitopeComplexDataset.FN_DATA_CONFIG = args.data_config
    ds = EpitopeComplexDataset.from_key(args.data, args=args)
    train_ds, val_ds = ds.train_test_split(test_size=0.2, shuffle=False)
    collator = TokenizingCollator(tokenizer=tokenizer,
                                  max_length=(ds.max_length + 2))  # +2 for [CLS] and [EOS]
    train_dataloader = DataLoader(train_ds,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.n_workers,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  collate_fn=collator)
    val_dataloader = DataLoader(val_ds,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.n_workers,
                                pin_memory=True,
                                persistent_workers=True,
                                collate_fn=collator)
    logger.info(f'Loaded the train and val dataset from {args.data} in {args.data_config}:'
                f'len(train_ds): {len(train_ds)}, len(val_ds): {len(val_ds)}')

    model = LangVAELitModel(lm=plm, r_factor=args.r_factor)
    logger.info(f'Model created: {model}')

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val.loss', mode='min', patience=args.patience))
    callbacks.append(ModelCheckpoint(dirpath=args.outdir,
                                     filename=args.filename,
                                     monitor='val.loss', mode='min',
                                     save_top_k=1))

    trainer = pl.Trainer(accelerator=args.accelerator,
                         devices=args.devices,
                         strategy=args.strategy,
                         # strategy=DeepSpeedStrategy(config=ds_config),
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         precision=args.precision,
                         max_epochs=args.max_epochs,
                         callbacks=callbacks,
                         # gradient_clip_val=1.0,
                         enable_checkpointing=True,
                         logger=CSVLogger(save_dir=args.outdir, name='logs'))

    # torch.set_autocast_enabled(True)
    logger.info(f'Start to train...')
    old_sd = copy.deepcopy(model.state_dict())
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    # Check the model.lm will not be trained

    new_sd = model.state_dict()
    for k, v in new_sd.items():
        if k.startswith('lm'):
            if not torch.equal(old_sd[k], v):
                logger.warning(f'>>> The model {k} has been trained during the training. <<<')
        else:
            if torch.equal(old_sd[k], v):
                logger.warning(f'>>> The model {k} has not been trained during the training. <<<')

    logger.info('Done to train.')


def gen_samples(args):
    logger.info(f'Start gen_samples with args: {args}')
    plm = AutoPeftModel.from_pretrained(args.plm_name_or_path, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.plm_name_or_path)
    model = LangVAELitModel.load_from_checkpoint(args.ckpt, lm=plm, r_factor=args.r_factor)
    logger.info(f'Loaded the model from {args.ckpt}: {model}')

    # Generate samples
    reconst_tokens = model.generate_samples(max_length=args.max_length, n=args.n_samples)
    seqs = tokenizer.batch_decode(reconst_tokens, skip_special_tokens=True)
    logger.info(f'Generated {len(seqs)} samples: {seqs}')


def main():
    parser = ArgumentParser('tcrvae')
    parser.add_argument('--log_level', type=str, default='DEBUG')
    subparsers = parser.add_subparsers()

    # Arguments for sub command 'gen_train_data'
    sub_parser = subparsers.add_parser('gen_train_data')
    sub_parser.set_defaults(func=gen_train_data)
    sub_parser.add_argument('--data_config', type=str, default='../config/data.json')
    sub_parser.add_argument('--data', type=str, default='tcrdb_pos,immunecode')
    sub_parser.add_argument('--n_workers', type=int, default=20)

    # Arguments for sub command 'create_peft_model'
    sub_parser = subparsers.add_parser('create_peft_model')
    sub_parser.set_defaults(func=create_peft_model)
    sub_parser.add_argument('--target_model', type=str, default='facebook/esm2_t33_650M_UR50D')
    sub_parser.add_argument('--outdir', type=str, default='../output/peft_esm2_t33_650M_UR50D')

    # Arguments for sub command 'run_train'
    sub_parser = subparsers.add_parser('run_train')
    sub_parser.set_defaults(func=run_train)
    sub_parser.add_argument('--data_config', type=str, default='../config/data.json')
    sub_parser.add_argument('--data', type=str, default='tcrdb_pos')
    sub_parser.add_argument('--plm_name_or_path', type=str, default='../output/peft_esm2_t33_650M_UR50D')
    sub_parser.add_argument('--r_factor', type=int, default=4)
    sub_parser.add_argument('--accelerator', type=str, default='gpu')
    sub_parser.add_argument('--devices', type=int, default=2)
    sub_parser.add_argument('--strategy', type=str, default='ddp_spawn')
    sub_parser.add_argument('--precision', type=int, default=16)
    sub_parser.add_argument('--batch_size', type=int, default=1024)
    sub_parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    sub_parser.add_argument('--n_workers', type=int, default=12)
    sub_parser.add_argument('--max_epochs', type=int, default=100)
    sub_parser.add_argument('--patience', type=int, default=10)
    sub_parser.add_argument('--outdir', type=str, default='../output/exp1')
    sub_parser.add_argument('--filename', type=str, default='vae_rf4')

    # Arguments for sub command 'gen_samples'
    sub_parser = subparsers.add_parser('gen_samples')
    sub_parser.set_defaults(func=gen_samples)
    sub_parser.add_argument('--plm_name_or_path', type=str, default='../output/peft_esm2_t33_650M_UR50D')
    sub_parser.add_argument('--ckpt', type=str, default='../output/exp1/vae_rf2.ckpt')
    sub_parser.add_argument('--n_samples', type=int, default=1000)
    sub_parser.add_argument('--max_length', type=int, default=15)
    sub_parser.add_argument('--r_factor', type=int, default=2)

    args = parser.parse_args()
    print(f'Logging level: {args.log_level}')
    logger.setLevel(args.log_level)
    args.func(args)


if __name__ == '__main__':
    main()
