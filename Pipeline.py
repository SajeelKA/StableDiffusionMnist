import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self, latent_dim=4, kernelSize=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernelSize, 2, 1),
            nn.ReLU(),
            nn.GroupNorm(2, 32),
            nn.Conv2d(32, 64, kernelSize, 2, 1),
            nn.ReLU(),
            nn.GroupNorm(2, 64)
        )
        self.mu = nn.Conv2d(64, latent_dim, kernelSize, padding=1)
        self.logvar = nn.Conv2d(64, latent_dim, kernelSize, padding=1)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernelSize, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernelSize, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernelSize, 2, 1),
            nn.Tanh()
        )
    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h) # B, 4, ...
    def reparam(self, mu, logvar):
        var = logvar.exp()
        stdev = var.sqrt()
        return mu + stdev * torch.randn_like(mu)
        # return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, -30, 20)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar, z

class Mini_UNet(nn.Module):
    def __init__(self, latent_dim, embed_dim):
        super().__init__()
        NUM_CLASSES = 10
        self.label_emb = nn.Embedding(NUM_CLASSES + 1, embed_dim)
        self.t_proj = nn.Linear(1, embed_dim)
        self.conv_in = nn.Conv2d(latent_dim + embed_dim*2, 64, 3, padding=1) #latent_dim *2 because t_emb and y_emb are same dim
        self.noise_out = nn.Sequential(
            nn.ReLU(),
            nn.GroupNorm(2, 64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(2, 64),
            nn.Conv2d(64, latent_dim, 3, padding=1)
        )
    def forward(self, x, t, y):
        B, _, H, W = x.shape
        t_emb = self.t_proj(t[:,None].float()/1000)[:,:,None,None].expand(B, -1, H, W)#.expand(-1, -1, H, W)
        # t_emb = self.t_proj(t[:,None].float())[:,:,None,None].expand(B, -1, H, W) #.expand(-1, -1, H, W)
        y_emb = self.label_emb(y)[:,:,None,None].expand(B, -1, H, W)
        inp = torch.cat([x, t_emb, y_emb], dim=1)

        return self.noise_out(self.conv_in(inp))

class Pipeline(nn.Module):
  def __init__(self, noise_channels, embed_channels, T):
    super().__init__()
    self.vae = VAE().to(device)
    self.unet = Mini_UNet(noise_channels, embed_channels).to(device)
    self.sample = Sampling((28,28), self.unet, self.vae, T)
    self.T = T

  def forward(self, img, labels):
    NULL_CLASS, guide_prob = 10, 0.1
    rec, mu, logvar, z = self.vae(img)
    reconstruction_loss = F.mse_loss(rec, img)
    kl = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
    vae_loss = reconstruction_loss + kl
    # Diffusion UNet forward+loss
    #z = self.vae.reparam(mu, logvar).detach()
    # z = self.vae.reparam(mu, logvar)
    z = z.detach()
    t = torch.randint(0, self.T,  (img.size(0),), device=device)
    noise = torch.randn_like(z)
    z_t = self.sample.q_sample(z, t, noise)
    y_in = labels.clone()
    mask = torch.rand_like(labels.float()) < guide_prob
    y_in[mask] = NULL_CLASS
    noise_pred = self.unet(z_t, t, y_in)
    diff_loss = F.mse_loss(noise_pred, noise)
    loss = diff_loss + vae_loss

    return loss, vae_loss

class Sampling():
  def __init__(self, input_shape, unet, vae, T):
    self.vae = vae
    self.unet = unet
    self.T = T
    self.input_shape = input_shape
    self.betas = torch.linspace(1e-4, 0.02,  self.T).to(device)
    self.alphas = 1 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, 0).to(device)

  def q_sample(self,x, t, noise):
      a = torch.sqrt(self.alphas_cumprod[t])[:,None,None,None]
      b = torch.sqrt(1 - self.alphas_cumprod[t])[:,None,None,None]
      return a * x + b * noise

  @torch.no_grad()
  def sample(self, label, scale=3.0, n=16):
      x_noisy = torch.randn(n, 4, self.input_shape[0], self.input_shape[1]).to(device)
      x_reduced_noise = x_noisy
      lab = torch.full((n,), label, device=device)
      t = self.T -1
      while t >= 0:
          x_noisy = x_reduced_noise
          tb = torch.full((n,), t, device=device)
          noise_conditional = self.unet(x_noisy, tb, lab)
          noise_unconditional = self.unet(x_noisy, tb, torch.full_like(lab, 10))
          noise_pred = noise_conditional + scale * (noise_conditional - noise_unconditional)
          a, a_prod, b = self.alphas[t], self.alphas_cumprod[t], self.betas[t]
          x_reduced_noise = (1/a.sqrt()) * (x_noisy - b / torch.sqrt(1 - a_prod) * noise_pred) #algorithm 2 line 4 in DDPM paper
          if t > 0:
              x_reduced_noise += torch.sqrt(b) * torch.randn_like(x_reduced_noise) #algorithm 2 line 4 in DDPM paper
          t -= 1

      return self.vae.decode(x_reduced_noise).cpu()


