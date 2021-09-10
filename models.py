import torch
import random
import pytorch_lightning as pl

from x_transformers import *
from x_transformers.autoregressive_wrapper import *

from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.vision_transformer import VisionTransformer
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from functools import partial

from utils import CosineAnnealingWarmUpRestarts

class TransformerOCR(pl.LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        backbone = ResNetV2(
        layers=cfg.backbone_layers, num_classes=0, global_pool='', in_chans=cfg.channels,
        preact=False, stem_type='same', conv_layer=StdConv2dSame)

        embed_layer = partial(HybridEmbed, backbone=backbone)

        self.encoder = CustomVisionTransformer( img_size=(cfg.max_height, cfg.max_width),
                                                patch_size=cfg.patch_size,
                                                in_chans=cfg.channels,
                                                num_classes=0,
                                                embed_dim=cfg.dim,
                                                depth=cfg.encoder_depth,
                                                num_heads=cfg.heads,
                                                embed_layer=embed_layer
                                                )
        self.decoder = CustomARWrapper(
                        TransformerWrapper(
                            num_tokens=len(tokenizer),
                            max_seq_len=cfg.max_seq_len,
                            attn_layers=Decoder(
                                dim=cfg.dim,
                                depth=cfg.num_layers,
                                heads=cfg.heads,
                                **cfg.decoder_cfg
                            )),
                        pad_value=cfg.pad_token
                    )
        self.bos_token = cfg.bos_token
        self.eos_token = cfg.eos_token
        self.max_seq_len = cfg.max_seq_len
        self.temperature = cfg.temperature

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.cfg.lr)

        if self.cfg.scheduler == "CosineAnnealingWarmUpRestarts":
            scheduler = CosineAnnealingWarmUpRestarts
            scheduler = {'scheduler': scheduler(optimizer,
                    T_0=self.cfg.T_0,
                    T_mult=int(self.cfg.T_mult),
                    eta_max=self.cfg.eta_max,
                    T_up=self.cfg.T_up,
                    gamma=self.cfg.gamma
                ),
                'interval': 'epoch',
                'name': self.cfg.scheduler}
        else:
            scheduler = getattr(torch.optim.lr_scheduler, self.cfg.scheduler)
            scheduler = {'scheduler': scheduler(optimizer, self.cfg.max_lr,
                total_steps=int(295805/self.cfg.batch_size*self.cfg.max_epochs)),
                     'interval': 'step',
                     'name': self.cfg.scheduler}
        return [optimizer], [scheduler]

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def forward(self, x):
        '''
        x: (B, C, W, H)
        labels: (B, S)

        # B : batch size
        # W : image width
        # H : image height
        # S : source sequence length
        # E : hidden size
        # V : vocab size
        '''

        encoded = self.encoder(x)
        dec = self.decoder.generate(torch.LongTensor([self.bos_token]*len(x))[:, None].to(x.device), self.max_seq_len,
                                    eos_token=self.eos_token, context=encoded, temperature=self.temperature)
        return dec

    def training_step(self, batch, batch_num):
        x, y = batch
        tgt_seq, tgt_mask = y
        encoded = self.encoder(x)
        loss = self.decoder(tgt_seq, mask=tgt_mask, context=encoded)

        return {'loss': loss}

    def validation_step(self, batch, batch_num):
        x, y = batch
        tgt_seq, tgt_mask = y
        encoded = self.encoder(x)
        loss = self.decoder(tgt_seq, mask=tgt_mask, context=encoded)
        dec = self.decoder.generate((torch.ones(x.size(0),1)*self.bos_token).long().to(x.device), self.max_seq_len,
                                    eos_token=self.eos_token, context=encoded, temperature=self.temperature)
        gt = self.tokenizer.decode(tgt_seq)
        pred = self.tokenizer.decode(dec)

        assert len(gt) == len(pred)

        acc = sum([1 if gt[i] == pred[i] else 0 for i in range(len(gt))]) / x.size(0)

        return {'val_loss': loss,
                'results' : {
                    'gt' : gt,
                    'pred' : pred
                    },
                'acc': acc
                }

    def validation_epoch_end(self, outputs):
        val_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        acc = sum([x['acc'] for x in outputs]) / len(outputs)

        wrong_cases = []
        for output in outputs:
            for i in range(len(output['results']['gt'])):
                gt = output['results']['gt'][i]
                pred = output['results']['pred'][i]
                if gt != pred:
                    wrong_cases.append("gt:{}~pred:{}".format(gt, pred))
        wrong_cases = random.sample(wrong_cases, 10)

        self.log('val_loss', val_loss)
        self.log('accuracy', acc)

        # custom text logging
        self.logger.log_text("wrong_case", " | ".join(wrong_cases), self.global_step)


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *cfg, **kwcfg):
        super(CustomARWrapper, self).__init__(*cfg, **kwcfg)

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwcfg):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwcfg.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            # print('arw:',out.shape)
            logits = self.net(x, mask=mask, **kwcfg)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is entmax:
                probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, *cfg, **kwcfg):
        super(CustomVisionTransformer, self).__init__(img_size=img_size, *cfg, **kwcfg)
        self.height, self.width = img_size

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x
