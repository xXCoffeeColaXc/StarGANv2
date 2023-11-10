from modules import Generator
from modules import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from utils import *
import wandb

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, model, config):
        self.model = model # contains dataloader, G, D and optimizers
        self.config = config # contains config data

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.model.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.model.d_optimizer.param_groups:
            param_group['lr'] = d_lr

        self.config.g_lr = g_lr # TODO test this
        self.config.d_lr = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.model.g_optimizer.zero_grad()
        self.model.d_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.config.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target) # This could be passed as CrossEntropy()

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.model.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.config.device)
        c_fixed_list = create_labels(c_org=c_org, c_dim=self.config.c_dim, config=self.config)

        # Learning rate cache for decaying.
        g_lr = self.config.g_lr
        d_lr = self.config.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.config.resume_iters:
            start_iters = self.config.resume_iters
            self.model.restore_config(self.config.resume_iters) # NOTE ?

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.config.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label2onehot(label_org, self.config.c_dim)
            c_trg = label2onehot(label_trg, self.config.c_dim)

            x_real = x_real.to(self.config.device)           # Input images.
            c_org = c_org.to(self.config.device)             # Original domain labels.
            c_trg = c_trg.to(self.config.device)             # Target domain labels.
            label_org = label_org.to(self.config.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.config.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.model.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # Compute loss with fake images.
            x_fake = self.model.G(x_real, c_trg)
            out_src, out_cls = self.model.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.config.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.model.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.config.lambda_cls * d_loss_cls + self.config.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.model.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/d_loss'] = d_loss.item()
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.config.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.model.G(x_real, c_trg)
                out_src, out_cls = self.model.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                # Target-to-original domain.
                x_reconst = self.model.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst)) # This is just an L1 loss
                # TODO try out feeding this loss to discriminator instead of generator,
                # cycle loss will be erased ? 

                # Backward and optimize.
                g_loss = g_loss_fake + self.config.lambda_rec * g_loss_rec + self.config.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.model.g_optimizer.step()

                # Logging.
                loss['G/g_loss'] = g_loss.item()
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.config.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.config.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
            
            # Translate fixed images for debugging.
            if (i+1) % self.config.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.model.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.config.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save config checkpoints.
            if (i+1) % self.config.model_save_step == 0:
                G_path = os.path.join(self.config.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.config.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.model.G.state_dict(), G_path)
                torch.save(self.model.D.state_dict(), D_path)
                print('Saved config checkpoints into {}...'.format(self.config.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.config.lr_update_step == 0 and (i+1) > (self.config.num_iters - self.config.num_iters_decay):
                g_lr -= (self.config.g_lr / float(self.config.num_iters_decay))
                d_lr -= (self.config.d_lr / float(self.config.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            # Log to wandb
            wandb.log(loss)
            wandb.log({
                "d_lr": self.config.d_lr,
                "g_lr": self.config.g_lr
            })
          
    def val(self):
        # Get a batch of validation data
        # NOTE just a batch, all do I need to iter through all of the val data?
        val_loader = self.model.val_loader
    
        total_val_loss = 0
        with torch.no_grad():
            for i, (x_real, label_org) in enumerate(val_loader)

                # Preprocess input
                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                c_org = label2onehot(label_org, self.config.c_dim)
                c_trg = label2onehot(label_trg, self.config.c_dim)

                x_real = x_real.to(self.config.device)           # Input images.
                c_org = c_org.to(self.config.device)             # Original domain labels.
                c_trg = c_trg.to(self.config.device)             # Target domain labels.
                label_org = label_org.to(self.config.device)     # Labels for computing classification loss.
                label_trg = label_trg.to(self.config.device)     # Labels for computing classification loss.


                # Compute losses
                x_fake = self.model.G(x_real, c_trg)
                out_src, out_cls = self.model.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                # Target-to-original domain.
                x_reconst = self.model.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.config.lambda_rec * g_loss_rec + self.config.lambda_cls * g_loss_cls

            # Aggregate val_loss through one batch
            total_val_loss += g_loss

        # Calculate mean
        mean_total_val_loss = total_val_loss / len(val_loader)

        # Log loss
        wandb.log({"val_loss": mean_total_val_loss})

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.model.restore_config(self.config.test_iters)
        
        data_loader = self.model.data_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.config.device)
                c_trg_list = create_labels(c_org=c_org, c_dim=self.config.c_dim, config=self.config)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.model.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.config.result_dir, '{}-images.jpg'.format(i+1))
                save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))