from google.colab import files
import time

import numpy as np
import torch
import torch.nn.functional as F
from utils import make_grid_show,threshold,expand_div
from pathenergy import PathEnergy


#the assumption is that the mass on the pass is not the same
def train(args, train_dataloader, my_autoencoder,optimizer, losses, num_epochs ):

  expand_div(args)  #the divergence operator is expanded, which is the core of unbalanced OTS
  
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

      #OT energy on  num_b pairs of images;
      #Since the batch is shuffled, we choose to interpolate bwtween img[i] and img[i+1] 
      #image_path keeps the originl decodered images, image_sequence_i have the clipped element and standard_mass  
      num_b =  min( args.N_Selected, image_batch.shape[0] ) - 1
      image_batch_down = F.interpolate(image_batch, size=(args.m, args.n),mode='bilinear')

      image_sequence_i = []
      image_path = []
      l1_ref = []
      for i in range(num_b):
          
          image_sequence_i.append( threshold(image_batch_down[i],args).to(torch.float64) ) #since the input is binary, so we need to normalize it
          image_path.append(image_batch_recon[i].unsqueeze(0) ) #default type is torch.float32
          l1_ref.append(image_batch[i].sum())
          for step in args.steps[1:args.T]:
              #print(step)
              code_sequence_i = image_batch_code[i] + step*(image_batch_code[i+1] - image_batch_code[i])
              recon_i = my_autoencoder.decoder(torch.unsqueeze(code_sequence_i,0))
              image_interpolated_i =  F.interpolate(recon_i, size=(args.m, args.n),mode='bilinear').squeeze(0)
              
              image_path.append(recon_i.clone())
              l1_ref.append( image_batch[i].sum() + step.item()*(image_batch[i+1].sum() - image_batch[i].sum())  )
              
              image_interpolated_i = image_interpolated_i.to(torch.float64) #torch32 will arise unstable linear system solution
              image_interpolated_i[:,args.obstacle] = 0 #cut the obstacle
              background_index = (image_interpolated_i<0.5) #clip
              #background_index = (image_interpolated_i<args.tol) #clip
              image_interpolated_i[background_index] = args.tol

              image_sequence_i.append(image_interpolated_i.clone())
              
      image_sequence_i.append(  threshold( image_batch_down[num_b] ,args).to(torch.float64) )
      image_path.append(image_batch_recon[num_b].unsqueeze(0))
      l1_ref.append(image_batch[num_b].sum())
      

      
      image_sequence_i = torch.cat( image_sequence_i )
      image_path = torch.cat(image_path)
      l1_ref = torch.stack(l1_ref)
      l1_ref.requires_grad = False     


      T_b = image_sequence_i.shape[0]-args.channel

      #define weight on stagger grid
      if args.imgcuroption == 'mid':
        image_cur = 0.5* image_sequence_i[0:T_b] + 0.5*image_sequence_i[args.channel:]
      else:
        image_cur = image_sequence_i[0:T_b]
        
      if args.boundary =='periodic':
        
        indices = [-1] + list(range(0, args.m-1)) #need to redefine it when m!=n
        image_j2 = 0.5*image_cur[:,:,indices] + 0.5*image_cur
        image_j1 = 0.5*image_cur[:,indices,:] + 0.5*image_cur
      else:
        zero_column = torch.zeros( (T_b, image_cur.shape[1],1), device=args.device )
        zero_row = torch.zeros( (T_b, 1, image_cur.shape[2]), device=args.device )
        image_j2 = 0.5*torch.cat( (image_cur, zero_column), dim=2 )  + 0.5*torch.cat( (zero_column,image_cur) , dim=2 )
        image_j1 = 0.5*torch.hstack ( (image_cur, zero_row)  ) + 0.5*torch.hstack ( (zero_row, image_cur)  ) 
      weight = torch.hstack( ( torch.flatten( image_j1, start_dim=1 ), torch.flatten( image_j2, start_dim=1 ) ) )
      #weight_c = torch.flatten( image_cur[:,~args.obstacle], start_dim=1 ) / args.tau #this is using the mid as the weight for source term
      weight_c = torch.flatten( image_sequence_i[:T_b,  ~args.obstacle], start_dim=1 ) / args.tau
      weight_expanded = torch.hstack( (weight[:,args.mask] ,weight_c) )

      #obtain partial_f
      image_sequence_i_time = image_sequence_i.reshape(-1,args.channel,args.m,args.n)
      image_diff = torch.diff( image_sequence_i_time, dim=0  )
      image_diff_v = -torch.reshape(image_diff, (T_b, args.m * args.n) )
      assert image_diff_v.dtype == torch.float64, "image_diff_v must be of dtype float64"     
      


      # loss function
      if args.bceloss:
        recon_error = F.binary_cross_entropy(image_batch_recon, image_batch)
      else:
        recon_error = F.mse_loss(image_batch_recon, image_batch)  
      l1_norm = torch.abs(image_path).sum(dim=(1,2,3))
      print(' current l1_norm: ',l1_norm.detach().cpu().numpy().tolist() )
      weight_error = F.l1_loss( l1_norm,   l1_ref)
      if num_b>0:
        t0 = time.time() 
        result_dict={}
        energy_term = PathEnergy.apply(  weight_expanded,image_diff_v[:,~args.obstacle.flatten()],args, result_dict) #/weight.shape[0]
        args.pathvariables = result_dict
        print(f"Energy term computation time used: {time.time()-t0:.2f} seconds")
        print('recon Loss:', recon_error.detach().item(), 'mass Loss: ', weight_error.detach().item(),  'energy regu: ', energy_term.detach().item() )
        loss = recon_error +  args.mass_weight* weight_error +   args.energy_weight* energy_term
        losses["mseterm"].append( recon_error.item()  )
        losses["massterm"].append( weight_error.item()  )
        losses["pathenergy"].append( energy_term.item() )
      

      
        
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

      #stack the source and show it
      Source = result_dict["source"].reshape(-1,args.m,args.n) 
      layer_max = Source.max(axis=(1, 2), keepdims=True)
      make_grid_show( torch.from_numpy(Source/layer_max), pad_value= 1 )

    if (epoch+1) % args.save_gap  == 0:
      name_current = 'model_' + args.savename + str(epoch) + '.pth'
      torch.save(my_autoencoder.state_dict(), name_current)
      files.download(name_current)
  
