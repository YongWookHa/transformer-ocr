import torch
from torch.nn.utils.rnn import pad_sequence
import albumentations as alb
import cv2
from albumentations.pytorch import ToTensorV2

from PIL import Image
from pathlib import Path


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, txt_fn):
        self.cfg = cfg
        self.images = []
        self.texts = []

        # build one
        self.token_id_dict = {
            "token2id": {
                "[PAD]": cfg.pad_token,
                "[BOS]": cfg.bos_token,
                "[EOS]": cfg.eos_token,
                "[OOV]": cfg.oov_token
            },
            "id2token": {
                cfg.pad_token: "[PAD]",
                cfg.bos_token: "[BOS]",
                cfg.eos_token: "[EOS]",
                cfg.oov_token: "[OOV]"
            }
        }

        skip_cnt, token_cnt = 0, 4
        with open(txt_fn, 'r', encoding='utf8') as f:
            for line in f:
                fn, text = line.strip().split('\t')
                if cfg.max_seq_len < len(text) + 2:
                    # we will add [BOS] and [EOS]
                    skip_cnt += 1
                    continue
                self.images.append(fn)
                self.texts.append(text)
                if not cfg.load_tokenizer:
                    for token in text:
                        if token not in self.token_id_dict["token2id"]:
                            self.token_id_dict["token2id"][token] = token_cnt
                            self.token_id_dict["id2token"][token_cnt] = token
                            token_cnt += 1

        print(f"{len(self.images)} data loaded. ({skip_cnt} data skipped)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Read the image (use PIL Image to load unicode name images)
        if cfg.channels == 1, need to change alb transform methods
        """
        idx = idx % len(self.images)
        image = cv2.imread(str(Path(self.cfg.image_dir) / self.images[idx]))
        text = self.texts[idx]
        return image, text

class CustomCollate(object):
    def __init__(self, cfg, tokenizer, is_train=True):
        self.cfg = cfg
        self.tokenizer = tokenizer

        if is_train:
            self.transform = alb.Compose(
                [
                    alb.Resize(cfg.max_height, cfg.max_width),
                    alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1,
                        border_mode=0, interpolation=3, value=[255, 255, 255], p=0.5),
                    alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                        value=[255, 255, 255], p=.5),
                    alb.GaussNoise(10, p=.2),
                    alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                    alb.JpegCompression(95, p=.3),
                    alb.ToGray(always_apply=True),
                    # TODO: Calculate normalize mean variance of training data
                    alb.Normalize(),
                    # alb.Sharpen()
                    alb.pytorch.ToTensorV2(),
                ]
            )
        else:
            self.transform = alb.Compose(
                [
                    alb.Resize(cfg.max_height, cfg.max_width),
                    alb.ToGray(always_apply=True),
                    alb.Normalize(),
                    # alb.Sharpen()
                    ToTensorV2(),
                ]
            )

    def __call__(self, batch):
        """
        return:
            images, (seq, mask)
        """
        np_images, texts = zip(*batch)
        images = []
        for img in np_images:
            images.append(self.transform(image=img)["image"])
        images = torch.stack(images)
        labels = self.tokenizer.encode(texts)

        return (images, labels)


class Tokenizer:
    def __init__(self, d):
        self.token2id = d["token2id"]
        self.id2token = d["id2token"]

    def __len__(self):
        return len(self.token2id)

    def encode(self, texts: list):
        """
        text:
            list of string form text
            [str, str, ...]
        return:
            tensors
        """
        pad = self.token2id["[PAD]"]
        bos = self.token2id["[BOS]"]
        eos = self.token2id["[EOS]"]
        oov = self.token2id["[OOV]"]

        ids = []
        for text in texts:
            encoded = [bos,]
            for token in text:
                try:
                    encoded.append(self.token2id[token])
                except KeyError:
                    encoded.append(oov)
            encoded.append(eos)
            ids.append(torch.tensor(encoded))

        seq = pad_sequence(ids, batch_first=True, padding_value=pad)
        mask = torch.zeros_like(seq)
        for i, encoded in enumerate(ids):
            mask[i, :len(encoded)] = 1

        return seq.long(), mask.bool()

    def decode(self, labels):
        """
        labels:
            [B, L] : B for batch size, L for Sequence Length
        """

        pad = self.token2id["[PAD]"]
        bos = self.token2id["[BOS]"]
        eos = self.token2id["[EOS]"]
        oov = self.token2id["[OOV]"]

        texts = []
        for label in labels.tolist():
            text = ""
            for id in label:
                if id == bos:
                    continue
                elif id == pad or id == eos:
                    break
                else:
                    text += self.id2token[id]

            texts.append(text)

        return texts
