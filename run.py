import torch
import pytorch_lightning as pl
import argparse

from pytorch_lightning.loggers import TensorBoardLogger
from models import TransformerOCR
from dataset import CustomDataset, CustomCollate, Tokenizer
from utils import load_setting, save_tokenizer

from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/debug.yaml",
                        help="Experiment settings")
    parser.add_argument("--version", "-v", type=int, default=0,
                        help="Train experiment version")
    parser.add_argument("--load_tokenizer", "-bt", type=str, default="",
                        help="Load pre-built tokenizer")
    parser.add_argument("--max_epochs", "-me", type=int, default=10,
                        help="Max epochs for training")
    parser.add_argument("--num_workers", "-nw", type=int, default=0,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int, default=4,
                        help="Batch size for training and validate")
    parser.add_argument("--resume_train", "-rt", type=str, default="",
                        help="Resume train from certain checkpoint")
    args = parser.parse_args()

    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)

    # ----- dataset -----
    train_set = CustomDataset(cfg, cfg.train_data)
    val_set = CustomDataset(cfg, cfg.val_data)
    tokenizer = Tokenizer(train_set.token_id_dict)

    save_path = "{}/{}.pkl".format(cfg.model_path, cfg.name.replace(' ', '_'))
    save_tokenizer(tokenizer, save_path)

    train_collate = CustomCollate(cfg, tokenizer)
    val_collate = CustomCollate(cfg, tokenizer)
    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers, collate_fn=train_collate)
    valid_dataloader = DataLoader(val_set, batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers, collate_fn=val_collate)

    model = TransformerOCR(cfg, tokenizer)

    if cfg.resume_train:
        model = model.load_from_checkpoint(cfg.resume_train)

    logger = TensorBoardLogger("tb_logs", name="model", version=cfg.version)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/version_{cfg.version}",
        filename="checkpoints-{epoch:02d}-{val_loss:.5f}",
        save_top_k=3,
        mode="min",
    )

    device_cnt = torch.cuda.device_count()
    trainer = pl.Trainer(gpus=device_cnt, max_epochs=cfg.max_epochs, logger=logger,
        num_sanity_val_steps=1, accelerator="ddp" if device_cnt > 1 else None,
        callbacks=[ckpt_callback,], resume_from_checkpoint=cfg.resume_train)

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)

    trainer = pl.Trainer()
