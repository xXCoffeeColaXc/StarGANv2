import os
import torch
from modules import Generator, Discriminator
import wandb

class Stargan(object):
    def __init__(self, config, loader) -> None:
        self.data_loader = loader
        self.config = config
        
        # Build the model and tensorboard.
        self.build_model()
        self.setup_logger()


    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.config.g_conv_dim, self.config.c_dim, self.config.g_repeat_num)
        self.D = Discriminator(self.config.image_size, self.config.d_conv_dim, self.config.c_dim, self.config.d_repeat_num) 

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.config.device)
        self.D.to(self.config.device)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.config.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.config.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def setup_logger(self):
            # Initialize WandB
            wandb.init(project='stargan-weather', entity='tamsyandro', config={
                "d_lr": self.config.d_lr,  # Both discriminator and generator learning rate
                "g_lr": self.config.g_lr,
                "num_iters": self.config.num_iters,
                "batch_size": self.config.batch_size,
                "image_size": self.config.image_size,
                "selected_domains": self.config.selected_attrs,
                "lambda_cls": self.config.lambda_cls,
                "lambda_rec": self.config.lambda_rec,
                "lambda_gp": self.config.lambda_gp,
                "n_critic": self.config.n_critic,
                "num_epoch_decay": self.config.num_iters_decay,
                "lr_update_step": self.config.lr_update_step,
                "d_depth": self.config.d_repeat_num,
                #"g_depth": config.G_SAMPLE_DEPTH,
                "g_bottleneck_depth": self.config.g_repeat_num,
                #"weight_init": config.WEIGHT_INIT,
                #"with_attention": False,
                "skip_connection": True,
                # ... Add other hyperparameters here
            })

            # Ensure DEVICE is tracked in WandB
            wandb.config.update({"device": self.config.device})