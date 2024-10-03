import argparse
import os

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from sunerf.data.loader.multi_instrument import MultiInstrumentDataModule
from sunerf.model.sunerf import save_state, PlasmaSuNeRFModule
from sunerf.train.callback import PlasmaImageCallback, AbsorptionCallback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    # setup paths
    base_path = config['base_path']
    os.makedirs(base_path, exist_ok=True)
    work_directory = config['work_directory'] if 'work_directory' in config else base_path
    os.makedirs(work_directory, exist_ok=True)

    # setup default configs
    data_config = config['data']
    model_config = config['model'] if 'model' in config else {}
    training_config = config['training'] if 'training' in config else {}
    logging_config = config['logging'] if 'logging' in config else {'project': 'sunerf'}

    # setup training config
    epochs = training_config['epochs'] if 'epochs' in training_config else 100
    log_every_n_steps = training_config['log_every_n_steps'] if 'log_every_n_steps' in training_config else None
    ckpt_path = training_config['meta_path'] if 'meta_path' in training_config else 'last'

    # initialize logger
    logger = WandbLogger(**logging_config, save_dir=work_directory)
    logger.experiment.config.update(config, allow_val_change=True)

    # initialize data module and model
    data_module = MultiInstrumentDataModule(**data_config, working_dir=work_directory)

    # initialize SuNeRF model
    sunerf = PlasmaSuNeRFModule(Rs_per_ds=data_module.Rs_per_ds, seconds_per_dt=data_module.seconds_per_dt,
                                validation_dataset_mapping=data_module.validation_dataset_mapping,
                                **model_config)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                          save_last=True,
                                          every_n_train_steps=log_every_n_steps)
    save_path = os.path.join(base_path, 'save_state.snf')
    save_callback = LambdaCallback(on_validation_end=lambda *args: save_state(sunerf, data_module, save_path))

    test_image_callback = PlasmaImageCallback(data_module.validation_dataset_mapping[0],
                                              data_module.config['image_shape'],
                                              cmaps=data_module.config['cmaps'])

    absorption_callback = AbsorptionCallback('absorption', data_module.validation_datasets['absorption'].image_shape)
    callbacks = [checkpoint_callback, save_callback, test_image_callback, absorption_callback]

    N_GPUS = torch.cuda.device_count()
    trainer = Trainer(max_epochs=epochs,
                      logger=logger,
                      devices=N_GPUS,
                      accelerator='gpu' if N_GPUS >= 1 else None,
                      strategy='dp' if N_GPUS > 1 else None,  # ddp breaks memory and wandb
                      num_sanity_val_steps=-1,  # validate all points to check the first image
                      val_check_interval=log_every_n_steps,
                      gradient_clip_val=0.5,
                      callbacks=callbacks)

    trainer.fit(sunerf, data_module, ckpt_path=ckpt_path)
    trainer.save_checkpoint(os.path.join(base_path, 'final.ckpt'))
