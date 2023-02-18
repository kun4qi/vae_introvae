import argparse
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from utils import load_json
from utils import check_manual_seed
from utils import Logger
from utils import ModelSaver
from utils import Time
from models import Encoder
from models import Decoder
from models.utils import apply_spectral_norm
from dataio import CKBrainMetDataModule
import functions.pytorch_ssim as pytorch_ssim


def calc_latent_dim(config):
    return (
        config.dataset.batch_size,
        config.model.z_dim,
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters))),
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters)))
    )
    

class introvae(LightningModule):
    def __init__(self, config, needs_save):
        super().__init__()
        self.config = config
        self.needs_save = needs_save
        self.output_dir = config.save.output_root_dir
        self.alpha = self.config.training.alpha
        self.beta = self.config.training.beta
        self.margin = self.config.training.margin
        self.batch_size = self.config.dataset.batch_size
        self.fixed_z = torch.randn(calc_latent_dim(self.config))
        self.automatic_optimization = False  # For advanced/expert users who want to do esoteric optimization schedules or techniques, use manual optimization.

        # networks
        self.E = Encoder(input_dim=self.config.model.input_dim, z_dim=self.config.model.z_dim, filters=self.config.model.enc_filters, activation=self.config.model.enc_activation).float()
        self.D = Decoder(input_dim=self.config.model.input_dim, z_dim=self.config.model.z_dim, filters=self.config.model.dec_filters, activation=self.config.model.dec_activation, final_activation=self.config.model.dec_final_activation).float()
        if config.model.enc_spectral_norm:
            apply_spectral_norm(self.E)
        if config.model.dec_spectral_norm:
            apply_spectral_norm(self.D)

    def l_recon(self, recon: torch.Tensor, target: torch.Tensor):
        if 'ssim' in self.config.training.loss:
            ssim_loss = pytorch_ssim.SSIM(window_size=11)

        if self.config.training.loss == 'l2':
            loss = F.mse_loss(recon, target, reduction='sum')

        elif self.config.training.loss == 'l1':
            loss = F.l1_loss(recon, target, reduction='sum')

        elif self.config.training.loss == 'ssim':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon)

        elif self.config.training.loss == 'ssim+l1':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon) \
                 + F.l1_loss(recon, target, reduction='sum')

        elif self.config.training.loss == 'ssim+l2':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon) \
                 + F.mse_loss(recon, target, reduction='sum')

        else:
            raise NotImplementedError

        return self.beta * loss / self.batch_size

    def l_reg(self, mu: torch.Tensor, log_var: torch.Tensor):
        loss = - 0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var))
        return loss / self.batch_size

    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            if self.needs_save:
                #save sampling and recon image
                if self.current_epoch == 1 or (self.current_epoch - 1) % self.config.save.save_image_epoch_interval == 0:
                    image = batch['image']
                    z, _, _ = self.E(image)
                    x_r = self.D(z)
                    x_p = self.D(self.fixed_z.to("cuda"))

                    #t-sne
                    if self.config.save.t_sne:
                        z = z.reshape(z.shape[0], -1).detach().cpu()
                        fixed_z = self.fixed_z.reshape(self.fixed_z.shape[0], -1).detach().cpu()
                        zs = torch.cat((z, fixed_z), 0)

                        tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
                        tsne_results = tsne.fit_transform(zs)

                        fig = plt.figure(figsize=(8, 4))
                        cmap = get_cmap("tab10")
                        labels = ['z', 'fixed_z']
                        for i in range(2):
                            indices = np.arange(i*self.config.dataset.batch_size, (i+1)*self.config.dataset.batch_size)
                            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=cmap(i), label=labels[i])
                        plt.legend(loc="upper left", fontsize=14)
                        plt.title(f"t-SNE")
                        fig.savefig(self.config.save.output_root_dir + self.config.save.study_name + f"/img_epoch{self.current_epoch}.png")

                    image = image.detach().cpu()
                    x_r = x_r.detach().cpu()
                    x_p = x_p.detach().cpu()

                    image = image[:self.config.save.n_save_images, ...]
                    x_r = x_r[:self.config.save.n_save_images, ...]
                    x_p = x_p[:self.config.save.n_save_images, ...]
                    self.logger.train_log_images(torch.cat([image, x_r, x_p]), self.current_epoch-1)

                    
        if self.needs_save:
            self.log('epoch', self.current_epoch, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            if self.global_step % 2 ==0:
                self.log('iteration', self.global_step/2, on_step=True, on_epoch=False, logger=True)
        
        e_optim, d_optim = self.optimizers()
        e_optim.zero_grad()
        d_optim.zero_grad()

        #Encoder training
        image = batch['image']
        z, z_mu, z_logvar = self.E(image)
        z_p = torch.randn_like(z)
        x_r = self.D(z)
        x_p = self.D(z_p)
        z_r, z_r_mu, z_r_logvar = self.E(x_r.detach())
        z_pp, z_pp_mu, z_pp_logvar = self.E(x_p.detach())
        l_enc_reg = self.l_reg(z_mu, z_logvar)
        l_enc_margin = self.alpha * (
            F.relu(self.margin - self.l_reg(z_r_mu, z_r_logvar)) +
            F.relu(self.margin - self.l_reg(z_pp_mu, z_pp_logvar))
        )
        l_enc_latent = self.config.training.w_enc_reg * l_enc_reg + self.config.training.w_enc_margin * l_enc_margin
        l_enc_recon = self.l_recon(x_r, image)
        l_enc_total = self.config.training.w_enc_latent * l_enc_latent + self.config.training.w_enc_recon * l_enc_recon 
        self.manual_backward(l_enc_total, retain_graph=True)
        e_optim.step()
        
        #Decoder training
        z, z_mu, z_logvar = self.E(image)
        z_p = torch.randn_like(z)
        x_r = self.D(z)
        x_p = self.D(z_p)
        z_r, z_r_mu, z_r_logvar = self.E(x_r)
        z_pp, z_pp_mu, z_pp_logvar = self.E(x_p)
        l_dec_latent = self.alpha * (self.l_reg(z_r_mu, z_r_logvar) + self.l_reg(z_pp_mu, z_pp_logvar))
        l_dec_recon = self.l_recon(x_r, image)
        l_dec_total = self.config.training.w_dec_latent * l_dec_latent + self.config.training.w_dec_recon * l_dec_recon
        self.manual_backward(l_dec_total)
        d_optim.step()

        if self.needs_save:
            self.log('E_TotalLoss', l_enc_total, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('E_EncodeLoss', l_enc_reg, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('E_MarginLoss', l_enc_margin, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('E_LatentLoss', l_enc_latent, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('E_ReconLoss', l_enc_recon, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('D_TotalLoss', l_dec_total, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('D_LatentLoss', l_dec_latent, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('D_ReconLoss', l_dec_recon, prog_bar=True, on_step=True, on_epoch=False, logger=True)
                
        return {'encoder_loss': l_enc_total, 'decoder_loss' : l_dec_total}
        

    def validation_step(self, batch, batch_idx):

        if self.config.training.val_mode == "train":
            for m in self.E.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

            for m in self.D.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

        #save sampling and recon image
        if batch_idx == 0:
            if self.needs_save:
                if self.current_epoch % self.config.save.save_image_epoch_interval == 0:
                    image = batch['image']
                    z, _, _ = self.E(image)
                    x_r = self.D(z)
                    x_p = self.D(self.fixed_z.to("cuda"))

                    if self.config.save.t_sne:
                        z = z.reshape(z.shape[0], -1).detach().cpu()
                        fixed_z = self.fixed_z.reshape(self.fixed_z.shape[0], -1).detach().cpu()
                        zs = torch.cat((z, fixed_z), 0)

                        tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
                        tsne_results = tsne.fit_transform(zs)

                        fig = plt.figure(figsize=(8, 4))
                        cmap = get_cmap("tab10")
                        labels = ['z', 'fixed_z']
                        for i in range(2):
                            indices = np.arange(i*self.config.dataset.batch_size, (i+1)*self.config.dataset.batch_size)
                            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=cmap(i), label=labels[i])
                        plt.legend(loc="upper left")
                        plt.title(f"t-SNE")
                        fig.savefig(self.config.save.output_root_dir + self.config.save.study_name + f"/val_img_epoch{self.current_epoch}.png")

                    image = image.detach().cpu()
                    x_r = x_r.detach().cpu()
                    x_p = x_p.detach().cpu()

                    image = image[:self.config.save.n_save_images, ...]
                    x_r = x_r[:self.config.save.n_save_images, ...]
                    x_p = x_p[:self.config.save.n_save_images, ...]
                    self.logger.val_log_images(torch.cat([image, x_r, x_p]), self.current_epoch)

        image = batch['image']
        z, z_mu, z_logvar = self.E(image)
        z_p = torch.randn_like(z)

        x_r = self.D(z)
        x_p = self.D(z_p)
        z_r, z_r_mu, z_r_logvar = self.E(x_r.detach())
        z_pp, z_pp_mu, z_pp_logvar = self.E(x_p.detach())

        l_enc_reg = self.l_reg(z_mu, z_logvar)
        l_enc_margin = self.alpha * (
            F.relu(self.margin - self.l_reg(z_r_mu, z_r_logvar)) +
            F.relu(self.margin - self.l_reg(z_pp_mu, z_pp_logvar))
        )
        l_enc_latent = self.config.training.w_enc_reg * l_enc_reg + self.config.training.w_enc_margin * l_enc_margin
        l_enc_recon = self.l_recon(x_r, image)
        l_enc_total = self.config.training.w_enc_latent * l_enc_latent + self.config.training.w_enc_recon * l_enc_recon 

        z, z_mu, z_logvar = self.E(image)
        z_p = torch.randn_like(z)
        x_r = self.D(z)
        x_p = self.D(z_p)

        z_r, z_r_mu, z_r_logvar = self.E(x_r)
        z_pp, z_pp_mu, z_pp_logvar = self.E(x_p)

        l_dec_latent = self.alpha * (self.l_reg(z_r_mu, z_r_logvar) + self.l_reg(z_pp_mu, z_pp_logvar))
        l_dec_recon = self.l_recon(x_r, image)
        l_dec_total = self.config.training.w_dec_latent * l_dec_latent + self.config.training.w_dec_recon * l_dec_recon

        if self.needs_save:
            metrics = {
            'epoch': self.current_epoch,
            'Val_E_TotalLoss': l_enc_total.item(),
            'Val_E_EncodeLoss': l_enc_reg.item(),
            'Val_E_MarginLoss': l_enc_margin.item(),
            'Val_E_LatentLoss': l_enc_latent.item(),
            'Val_E_ReconLoss': l_enc_recon.item(),
            'Val_D_TotalLoss': l_dec_total.item(),
            'Val_D_LatentLoss': l_dec_latent.item(),
            'Val_D_ReconLoss': l_dec_recon.item()
            }
            self.logger.log_val_metrics(metrics)
                
        return {'encoder_loss': l_enc_total, 'decoder_loss' : l_dec_total}


    def configure_optimizers(self):
        e_optim = optim.Adam(filter(lambda p: p.requires_grad, self.E.parameters()), self.config.optimizer.enc_lr, [0.9, 0.9999])
        d_optim = optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.config.optimizer.dec_lr, [0.9, 0.9999])
        return [e_optim, d_optim]


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    #set logger
    monitoring_metrics = ['epoch', 'iteration', 'E_TotalLoss', 'E_EncodeLoss', 'E_MarginLoss',
                          'E_LatentLoss', 'E_ReconLoss', 'D_TotalLoss', 'D_LatentLoss', 'D_ReconLoss',
                          'Val_E_TotalLoss', 'Val_E_EncodeLoss', 'Val_E_MarginLoss',
                          'Val_E_LatentLoss', 'Val_E_ReconLoss', 'Val_D_TotalLoss', 'Val_D_LatentLoss', 'Val_D_ReconLoss']
  
    logger = Logger(save_dir=config.save.output_root_dir,
                    config=config,
                    seed=config.training.seed,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics
                    )
    save_dir_path = logger.log_dir
    os.makedirs(save_dir_path, exist_ok=True)
    
    #save config
    logger.log_hyperparams(config, needs_save)

    #set callbacks
    checkpoint_callback = ModelSaver(
        limit_num=config.save.n_saved,
        save_interval=config.save.save_epoch_interval,
        monitor=None,
        dirpath=logger.log_dir,
        filename='ckpt-{epoch:04d}',
        save_top_k=-1,
        save_last=False
    )

    #time per epoch
    timer = Time(config)

    dm = CKBrainMetDataModule(config)

    trainer = Trainer(
        default_root_dir=config.save.output_root_dir,
        accelerator="gpu",
        devices=config.training.visible_devices,
        strategy=DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=config.training.sync_batchnorm,
        max_epochs=config.training.n_epochs,
        callbacks=[checkpoint_callback, timer],
        logger=logger,
        deterministic=False,
        log_every_n_steps=1,
        num_sanity_val_steps = 0
        )
    
    dm.prepare_data()
    dm.setup(stage="fit")
    print('Training starts: [max_epochs] {}, [max_iterations] {}'.format(
        config.training.n_epochs, config.training.n_epochs * len(dm.train_dataloader()))
    )

    if not config.model.saved:
      model = introvae(config, needs_save, *dm.size())
      trainer.fit(model, dm)

    else:
      print(f'model load from {config.save.load_model_dir + config.save.model_savename}')
      model = introvae.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config, needs_save=needs_save,)
      trainer.fit(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)
