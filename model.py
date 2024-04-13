import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                    #from channel to depth 16
                    nn.Conv2d(1,16,3,padding='same' ),

                    #first block
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),

                    #second
                    nn.Conv2d(16,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),

                    #third
                    nn.Conv2d(32,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),



                    #256 --> latent16
                    nn.Conv2d(64,256,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256,16,3,padding='same' ),
                )
        
        self.decoder = nn.Sequential(
                    

                    #first block
                    nn.Conv2d(16,64,3,padding='same'),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #second block
                    nn.Conv2d(64,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #third block
                    nn.Conv2d(32,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,1,3,padding='same' ),
                )
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon

#ADD sigmoid layer at the end of AE to encourage binary output        
class Autoencoder_s_rgb(nn.Module):
    def __init__(self):
        super(Autoencoder_s_rgb, self).__init__()
        self.encoder = nn.Sequential(
                    #from channel to depth 16
                    nn.Conv2d(3,16,3,padding='same' ),

                    #first block
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),

                    #second
                    nn.Conv2d(16,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),

                    #third
                    nn.Conv2d(32,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),



                    #256 --> latent16
                    nn.Conv2d(64,256,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256,16,3,padding='same' ),
                )
        
        self.decoder = nn.Sequential(
                    

                    #first block
                    nn.Conv2d(16,64,3,padding='same'),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #second block
                    nn.Conv2d(64,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #third block
                    nn.Conv2d(32,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,3,3,padding='same' ),
                    nn.Sigmoid(),
                )
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon


#ADD sigmoid layer at the end of AE to encourage binary output        
class Autoencoder_s(nn.Module):
    def __init__(self):
        super(Autoencoder_s, self).__init__()
        self.encoder = nn.Sequential(
                    #from channel to depth 16
                    nn.Conv2d(1,16,3,padding='same' ),

                    #first block
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),

                    #second
                    nn.Conv2d(16,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),

                    #third
                    nn.Conv2d(32,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),



                    #256 --> latent16
                    nn.Conv2d(64,256,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256,16,3,padding='same' ),
                )
        
        self.decoder = nn.Sequential(
                    

                    #first block
                    nn.Conv2d(16,64,3,padding='same'),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #second block
                    nn.Conv2d(64,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #third block
                    nn.Conv2d(32,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,1,3,padding='same' ),
                    nn.Sigmoid(),
                )
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon


#ADD sigmoid layer at the end of AE to encourage binary output        
class Autoencoder_4block_s(nn.Module):
    def __init__(self,latent_channel):
        super(Autoencoder_4block_s, self).__init__()
        self.encoder = nn.Sequential(
                    #from channel to depth 16
                    nn.Conv2d(1,16,3,padding='same' ),

                    #first block
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),

                    #second
                    nn.Conv2d(16,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),

                    #third
                    nn.Conv2d(32,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),
                    
                    #fourth
                    nn.Conv2d(64,128,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(128,128,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),



                    #256 --> latent16
                    nn.Conv2d(128,256,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256,latent_channel,3,padding='same' ),
                )
        
        self.decoder = nn.Sequential(
                    
                    #first block
                    nn.Conv2d(latent_channel,128,3,padding='same'),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(128,128,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #first block
                    nn.Conv2d(128,64,3,padding='same'),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #second block
                    nn.Conv2d(64,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #third block
                    nn.Conv2d(32,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,1,3,padding='same' ),
                    nn.Sigmoid(),
                )
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon


#ADD relu layer at the end        
class Autoencoder_l(nn.Module):
    def __init__(self):
        super(Autoencoder_l, self).__init__()
        self.encoder = nn.Sequential(
                    #from channel to depth 16
                    nn.Conv2d(1,16,3,padding='same' ),

                    #first block
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),

                    #second
                    nn.Conv2d(16,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),

                    #third
                    nn.Conv2d(32,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2,2),



                    #256 --> latent16
                    nn.Conv2d(64,256,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256,16,3,padding='same' ),
                )
        
        self.decoder = nn.Sequential(
                    

                    #first block
                    nn.Conv2d(16,64,3,padding='same'),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64,64,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #second block
                    nn.Conv2d(64,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32,32,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    #third block
                    nn.Conv2d(32,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),

                    nn.Conv2d(16,16,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16,1,3,padding='same' ),
                    nn.LeakyReLU(0.2),
                )
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon