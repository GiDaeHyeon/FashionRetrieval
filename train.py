import pytorch_lightning as plfrom pytorch_lightning.loggers import TensorBoardLoggerfrom pytorch_lightning.callbacks.early_stopping import EarlyStoppingfrom pytorch_lightning.callbacks import ModelCheckpointfrom pytorch_lightning.plugins import DDPPluginimport warningswarnings.filterwarnings("ignore")from config import *import argparseparser = argparse.ArgumentParser(description="which network will you use")parser.add_argument('--target')args = parser.parse_args()target = args.targetif target == 'segmentation':    from network.segmentation import SegmentationNetwork    from dataloader.segmentation import SegmentationDataModule    net = SegmentationNetwork()    dm = SegmentationDataModule()elif target == 'triplet':    from network.triplet import TripletNetwork    net = TripletNetwork()logger = TensorBoardLogger(                           save_dir=f"{target}_log",                           name=f'{target}_{VERSION}',                           default_hp_metric=False,                           )checkpoint_callback = ModelCheckpoint(                                      monitor='val_loss',                                      dirpath=CKPT_DIR,                                      filename=CKPT_NAME,                                      mode='min'                                     )early_stop_callback = EarlyStopping(                                    monitor='val_loss',                                    min_delta=1e-4,                                    patience=10,                                    verbose=True,                                    mode='min'                                    )trainer = pl.Trainer(                     max_epochs=MAX_EPOCH,                     logger=logger,                     num_sanity_val_steps=0,                     accelerator='ddp',                     gpus=GPUS,                     plugins=DDPPlugin(find_unused_parameters=False),                     callbacks=[early_stop_callback,                                checkpoint_callback]                     )if __name__ == '__main__':    trainer.fit(net, datamodule=dm)