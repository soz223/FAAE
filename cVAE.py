
import torch.nn as nn
import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Parameter

def compute_ll(x, x_recon):
    return x_recon.log_prob(x).sum(1, keepdims=True).mean(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, alpha_focal, gamma_focal, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        if targets[0].item() == 0:
            F_loss = self.alpha_focal * (1-pt)**self.gamma_focal * BCE_loss
        else:
            F_loss = (1-self.alpha_focal) * pt**self.gamma_focal * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class Encoder(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim, 
                c_dim,
                non_linear=False):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_dim
        self.z_dim = hidden_dim[-1]
        self.c_dim = c_dim
        self.non_linear = non_linear
        self.layer_sizes_encoder = [input_dim + c_dim] + self.hidden_dims
        lin_layers = [nn.Linear(dim0, dim1, bias=True) for dim0, dim1 in zip(self.layer_sizes_encoder[:-1], self.layer_sizes_encoder[1:])]
               
        self.encoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.enc_mean_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True)
        self.enc_logvar_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True)

    def forward(self, x, c):
        # print(c.shape)
        # print(self.c_dim)
        c = c.reshape(-1, self.c_dim)

        h1 = torch.cat((x, c), dim=1)
        for it_layer, layer in enumerate(self.encoder_layers):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.leaky_relu(h1)

        mu = self.enc_mean_layer(h1)
        logvar = self.enc_logvar_layer(h1)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim,
                c_dim,
                non_linear=False, 
                init_logvar=-3):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dims = hidden_dim[::-1]
        self.non_linear = non_linear
        self.init_logvar = init_logvar
        self.c_dim = c_dim
        self.layer_sizes_decoder = self.hidden_dims + [input_dim]
        self.layer_sizes_decoder[0] = self.hidden_dims[0] + c_dim
        lin_layers = [nn.Linear(dim0, dim1, bias=True) for dim0, dim1 in zip(self.layer_sizes_decoder[:-1], self.layer_sizes_decoder[1:])]
        self.decoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.decoder_mean_layer = nn.Linear(self.layer_sizes_decoder[-2],self.layer_sizes_decoder[-1], bias=True)
        tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(self.init_logvar)
        self.logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)


    def forward(self, z, c):
        c = c.reshape(-1, self.c_dim)
        x_rec = torch.cat((z, c),dim=1)
        for it_layer, layer in enumerate(self.decoder_layers):
            x_rec = layer(x_rec)
            if self.non_linear:
                x_rec = F.leaky_relu(x_rec)

        mu_out = self.decoder_mean_layer(x_rec)
        return Normal(loc=mu_out, scale=self.logvar_out.exp().pow(0.5))
 

    
class Discriminator(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim,
                c_dim,
                non_linear=False, 
                init_logvar=-3):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dims = hidden_dim[::-1]
        self.non_linear = non_linear
        #self.c_dim = c_dim
        self.layer_sizes_discriminator = self.hidden_dims + [1]
        self.layer_sizes_discriminator[0] = self.hidden_dims[0] 
        lin_layers = [nn.Linear(dim0, dim1, bias=True) for dim0, dim1 in zip(self.layer_sizes_discriminator[:-1], self.layer_sizes_discriminator[1:])]
        self.discriminator_layers = nn.Sequential(*lin_layers[0:-1])
        self.discriminator_mean_layer = nn.Linear(self.layer_sizes_discriminator[-2],self.layer_sizes_discriminator[-1], bias=True)

    def forward(self, z):
        x_pred = z
        for it_layer, layer in enumerate(self.discriminator_layers):
            x_pred = layer(x_pred)
            if self.non_linear:
                x_pred = F.leaky_relu(x_pred)

        mu_out = self.discriminator_mean_layer(x_pred)
        return mu_out   
    
    

class cVAE(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                latent_dim,
                c_dim, 
                learning_rate=0.0001, 
                non_linear=False):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.learning_rate = learning_rate
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear)
        self.decoder = Decoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) 
        self.discriminator = Discriminator(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) 
        self.optimizer1 = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate) 
        self.optimizer2 = optim.Adam(list(self.discriminator.parameters()), lr=self.learning_rate) 
        self.optimizer3 = optim.Adam(list(self.encoder.parameters()), lr=self.learning_rate) 
    
    def encode(self, x, c):
        return self.encoder(x, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z, c):
        return self.decoder(z, c)
    
    def discriminat(self, z):
        return self.discriminator(z)

    def calc_kl(self, mu, logvar):
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def forward(self, x, c):
        self.zero_grad()
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z, c)
        fwd_rtn = {'x_recon': x_recon,
                    'mu': mu,
                    'logvar': logvar}
        return fwd_rtn
    
    def forward2(self, x, c, z_dim):
        self.zero_grad()
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        dc_fake = self.discriminat(z)
        real_distribution = torch.normal(mean=0.0,std=1.0,size=(x.shape[0],z_dim)).to(DEVICE)
        dc_real = self.discriminat(real_distribution)
        fwd_rtn = {'dc_fake': dc_fake,
                   'dc_real': dc_real}
        return fwd_rtn
    
    def forward3(self, x, c):
        self.zero_grad()
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        dc_fake = self.discriminat(z)
        fwd_rtn = {'dc_fake': dc_fake}
        return fwd_rtn

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn['x_recon']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']

        kl = self.calc_kl(mu, logvar)
        recon = self.calc_ll(x, x_recon)

        total = kl - recon
        losses = {'total': total,
                'kl': kl,
                'll': recon}
    
        return losses
    
    # # focal loss and BCE loss regularisation for the discriminator
    # def loss_function2(self, x, fwd_rtn, alpha_focal, gamma_focal, lambda_reg=0, logits=True, reduction='mean'):
    #     if alpha_focal == 0:
    #         loss_focal = FocalLoss(alpha_focal=alpha_focal, gamma_focal=gamma_focal, logits=True, reduction='mean')  # logits=True because we are applying it before the sigmoid activation
    #         loss_BCE = nn.BCEWithLogitsLoss()        
    #     else:
    #         loss_focal = FocalLoss(alpha_focal=alpha_focal, gamma_focal=gamma_focal, logits=True, reduction='mean')  # logits=True because we are applying it before the sigmoid activation
    #         loss_BCE = nn.BCEWithLogitsLoss()
    #     real_output = fwd_rtn['dc_real']
    #     fake_output = fwd_rtn['dc_fake']

    #     loss_focal_real = loss_focal(real_output, torch.ones_like(real_output))
    #     loss_focal_fake = loss_focal(fake_output, torch.zeros_like(fake_output))
    #     dc_loss_focal = loss_focal_real + loss_focal_fake  # 0*loss_real because we don't want to train the discriminator on real data

    #     loss_BCE_real = loss_BCE(real_output, torch.ones_like(real_output))
    #     loss_BCE_fake = loss_BCE(fake_output, torch.zeros_like(fake_output))
    #     dc_loss_BCE = loss_BCE_real + loss_BCE_fake  # 0*loss_real because we don't want to train the discriminator on real data

    #     # regulzarisation term
    #     dc_loss = dc_loss_focal + lambda_reg * dc_loss_BCE


    #     losses = {'dc_loss':dc_loss}     
    #     return losses


    # focal loss only for the discriminator
    def loss_function2(self, x, fwd_rtn, alpha_focal, gamma_focal, lambda_reg=0, logits=True, reduction='mean'):
        if alpha_focal == 0:
            loss = nn.BCEWithLogitsLoss()
        else:
            loss = FocalLoss(alpha_focal=alpha_focal, gamma_focal=gamma_focal, logits=True, reduction='mean')  # logits=True because we are applying it before the sigmoid activation
        real_output = fwd_rtn['dc_real']
        fake_output = fwd_rtn['dc_fake']

        # print('alpha_focal: ', alpha_focal)
        # print('gamma_focal: ', gamma_focal)

        loss_real = loss(real_output, torch.ones_like(real_output))
        # print('loss_real: ', loss_real)
        loss_fake = loss(fake_output, torch.zeros_like(fake_output))
        # print('loss_fake: ', loss_fake)
        dc_loss = loss_real + loss_fake  # 0*loss_real because we don't want to train the discriminator on real data
        if alpha_focal == 0:
            dc_loss = 0*loss_real + loss_fake
        # dc_loss = 0
        losses = {'dc_loss':dc_loss}     
        return losses


    # # BCE loss only for the discriminator
    # def loss_function2(self, x, fwd_rtn, alpha_focal, gamma_focal, logits=True, reduction='mean'):
    #     loss = nn.BCEWithLogitsLoss()
    #     real_output = fwd_rtn['dc_real']
    #     fake_output = fwd_rtn['dc_fake']
        
    #     loss_real = loss(real_output, torch.ones_like(real_output))
    #     loss_fake = loss(fake_output, torch.zeros_like(fake_output))
    #     dc_loss = 0*loss_real + loss_fake
    #     losses = {'dc_loss':dc_loss}     
    #     return losses

    def loss_function3(self, x, fwd_rtn):
        loss = nn.BCEWithLogitsLoss()
        fake_output = fwd_rtn['dc_fake']    
        gen_loss = loss(fake_output, torch.ones_like(fake_output))
        losses = {'gen_loss':gen_loss}     
        return losses
    
    
    def pred_latent(self, x, c, DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu, logvar = self.encode(x, c)   
        latent = mu.cpu().detach().numpy()
        latent_var = logvar.exp().cpu().detach().numpy()
        return latent, latent_var

    def pred_recon(self, x, c,  DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu, _ = self.encode(x, c)
            x_pred = self.decode(mu, c).loc.cpu().detach().numpy()
        return x_pred
    
    def pred_recon_tensor(self, x,c, test_latent, test_var, DEVICE):
        x = torch.FloatTensor(x.numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            x_pred = self.decode(torch.from_numpy(test_latent),c).loc.cpu().detach().numpy()
        return x_pred