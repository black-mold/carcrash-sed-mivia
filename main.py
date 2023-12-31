import argparse
import yaml
import torch
import sys
import importlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import lightning_fabric as lf

from engine import FeatureExtractor


## Parse arguments
parser = argparse.ArgumentParser(description = "Speaker verification with sequential module")

parser.add_argument('--config',         type=str,   default='./configs/mnist.yaml',   help='Config YAML file')
parser.add_argument('--mode',         type=str,   default='train',   help='choose train/val/eval')

args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)


print(args)
print(config)
print('Python Version:', sys.version)
print('PyTorch Version:', torch.__version__)
print('Number of GPUs:', torch.cuda.device_count())





def train():
    # sets seeds for numpy, torch and python.random.    
    lf.utilities.seed.seed_everything(seed = config['random_seed'])

    # ⚡⚡ 1. Set 'Dataset', 'DataLoader'

    training_dataset = importlib.import_module('dataloader.' + config['TRAIN_DATASET']).__getattribute__("MiviaDataset")
    training_dataset = training_dataset(**config['TRAIN_DATASET_CONFIG'])
    validation_dataset = importlib.import_module('dataloader.' + config['VAL_DATASET']).__getattribute__("MiviaDataset")
    validation_dataset = validation_dataset(**config['VAL_DATASET_CONFIG'])

    train_dataloader = DataLoader(
            dataset = training_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True
        )

    validation_dataloader = DataLoader(
            dataset = validation_dataset,
            batch_size=1,
            num_workers=config['num_workers'],
            pin_memory=True
        )


    # ⚡⚡ 2. Set 'Model', 'Loss', 'Optimizer', 'Scheduler'
    preprocess = importlib.import_module('models.' + config['feature_extractor']).__getattribute__("feature_extractor")
    preprocess =  preprocess(**config['feature_extractor_config'])

    model = importlib.import_module('models.' + config['model']).__getattribute__("MainModel")
    model =  model(**config['model_config'])

    optimizer = importlib.import_module("optimizer." + config['optimizer']).__getattribute__("Optimizer")
    optimizer = optimizer(model.parameters(), **config['optimizer_config'])

    loss_function = importlib.import_module("loss." + config['loss']).__getattribute__("loss_function")

    scheduler = importlib.import_module("scheduler." + config['scheduler']).__getattribute__("Scheduler")
    scheduler = scheduler(optimizer, **config['scheduler_config'])


    # ⚡⚡  3. Set 'engine' for training/validation and 'Trainer' 
    engine = FeatureExtractor(preprocess = preprocess, model = model, optimizer=optimizer, loss_function=loss_function, scheduler=scheduler) 

    # resume training from an old checkpoint
    # if config['resume_checkpoint']  is not None:
    #     feature_extractor = feature_extractor.load_from_checkpoint(model = model, optimizer=optimizer, loss_function=loss_function, scheduler=scheduler, checkpoint_path = config['resume_checkpoint'])
    #     print(config['resume_checkpoint'] + "are loaded")


    # ⚡⚡ 4. Init ModelCheckpoint callback, monitoring "val_ACC"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="avg_validation_f1_score",
        mode="max",
        filename="sample-mnist-{epoch:02d}-{avg_validation_f1_score:.2f}-{avg_validation_threshold:.2f}",
    )

    # ⚡⚡ 5. LightningModule
    trainer = pl.Trainer(
        deterministic=True, # Might make your system slower, but ensures reproducibility.
        default_root_dir = config['default_root_dir'], #
        devices = config['devices'], #
        val_check_interval = 1.0, # Check val every n train epochs.
        max_epochs = config['max_epoch'], #
        auto_lr_find = True, # ⚡⚡
        sync_batchnorm = True, # ⚡⚡
        callbacks = [checkpoint_callback], #
        accelerator = config['accelerator'], #
        num_sanity_val_steps = config['num_sanity_val_steps'], # Sanity check runs n batches of val before starting the training routine. This catches any bugs in your validation without having to wait for the first validation check. 
        replace_sampler_ddp = False, # ⚡⚡
        gradient_clip_val=1.0, # ⚡⚡
    )

    # ⚡⚡ 6. Resume training

    if config['resume_checkpoint']  is not None:
        trainer.fit(engine, train_dataloader, validation_dataloader, ckpt_path=config['resume_checkpoint'])
        print(config['resume_checkpoint'] + "are loaded")
    else:
        trainer.fit(engine, train_dataloader, validation_dataloader)
        print("no pre-trained weight are loaded")


def test():
    print("test")

    # ⚡⚡ 1. Set 'Dataset', 'DataLoader'
    test_dataset = importlib.import_module('dataloader.' + config['TEST_DATASET']).__getattribute__("MiviaDataset")
    test_dataset = test_dataset(**config['TEST_DATASET_CONFIG'])


    test_dataloader = DataLoader(
            dataset = test_dataset,
            batch_size=1,
            num_workers=config['num_workers'],
            pin_memory=True
        )


    # ⚡⚡ 2. Set 'Model', 'Loss', 'Optimizer', 'Scheduler'
    # Note that 'Optimizer' and 'Scheduler' are not needed for testing. They can be set to None.
    preprocess = importlib.import_module('models.' + config['feature_extractor']).__getattribute__("feature_extractor")
    preprocess =  preprocess(**config['feature_extractor_config'])

    model = importlib.import_module('models.' + config['model']).__getattribute__("MainModel")
    model =  model(**config['model_config'])

    optimizer = importlib.import_module("optimizer." + config['optimizer']).__getattribute__("Optimizer")
    optimizer = optimizer(model.parameters(), **config['optimizer_config'])

    loss_function = importlib.import_module("loss." + config['loss']).__getattribute__("loss_function")

    scheduler = importlib.import_module("scheduler." + config['scheduler']).__getattribute__("Scheduler")
    scheduler = scheduler(optimizer, **config['scheduler_config'])



    # ⚡⚡  3. Load model
    feature_extractor = FeatureExtractor.load_from_checkpoint(model = model, optimizer=optimizer, loss_function=loss_function, scheduler=scheduler, checkpoint_path = config['resume_checkpoint'])

    # ⚡⚡ 4. LightningModule
    trainer = pl.Trainer(accelerator=config['accelerator'], gpus = config['devices'])

    trainer.test(feature_extractor, dataloaders=test_dataloader)

    
if __name__ == "__main__":

    if args.mode == "train":
        train()

    elif args.mode == "test":
        test()

    # sets seeds for numpy, torch and python.random.
    