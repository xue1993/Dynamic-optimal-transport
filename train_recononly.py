from google.colab import files
import time

import numpy as np
import torch
import torch.nn.functional as F
from utils import make_grid_show,normalize_threshold
from pathenergy import PathEnergy


#the assumption is that the mass on the pass ARE SAME
def train(args, train_dataloader, my_autoencoder,optimizer, losses, num_epochs ):


  
  print('Training start...')
  torch.autograd.set_detect_anomaly(True)

  for epoch in range(num_epochs):
    losses["train_loss_avg"].append(0)
    num_batches = 0
    
    for image_batch in train_dataloader:

      #image_batch = train_data         
      image_batch = image_batch[0].to(args.device)        
      image_batch_code = my_autoencoder.encoder(image_batch)
      image_batch_recon = my_autoencoder(image_batch)
      image_batch_recon_code = my_autoencoder.encoder(image_batch_recon)

      

      # loss function
      if args.bceloss:
        recon_error = F.binary_cross_entropy(image_batch_recon, image_batch)
      else:
        recon_error = F.mse_loss(image_batch_recon, image_batch)  

      loss = recon_error

      
        
      # backpropagation
      optimizer.zero_grad()
      loss.backward()
      
      optimizer.step()
      
      losses["train_loss_avg"][-1] += loss.item()
      num_batches += 1

    losses["train_loss_avg"][-1] /= num_batches
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, losses["train_loss_avg"][-1]))


    if (epoch+1) % args.imshow_gap == 0:
      make_grid_show( image_path, pad_value= args.pad_value ) # image_sequence_i
      make_grid_show( image_sequence_i_time, pad_value= args.pad_value )


    if (epoch+1) % args.save_gap  == 0:
      name_current = 'model_' + str(epoch) + '.pth'
      torch.save(my_autoencoder.state_dict(), name_current)
      files.download(name_current)
  
