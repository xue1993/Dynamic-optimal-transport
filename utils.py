import numpy as np
import torch

#functions used on torchvision
from torchvision.utils import make_grid
from torchvision import datasets, transforms

from PIL import Image
import matplotlib.pyplot as plt
from scipy import sparse as sp

import pickle
from google.colab import files

#############for training(need to check if I need to discard one)##################
def normalize_threshold(X, args):
  X = X.clone().detach()    
  X[:,args.obstacle] = 0 #in case X[args.obstacle] is too large
  for i in range(X.shape[0]):      
    background_index = (X[i]<args.tol) #clip
    #normalize
    background_num = float(background_index.sum().item())
    X[i,background_index] = args.tol
    X[i,~background_index] *= (args.mass_standard-args.tol* background_num) / X[i,~background_index].sum()
  return X

#this will work on the training data, so clone().detach() doesn't affect much
def threshold(X, args):
  X = X.clone().detach()    
  X[:,args.obstacle] = 0
  background_index = (X<args.tol) #clip
  X[background_index] = args.tol  #X[args.obstacle] is also thresholded to be tol value
  return X

#######################for visualization#####################

def visualize_path(args, train_dataloader, my_autoencoder, T=None, flip_ = False):
  steps = (torch.arange(T+1, device=args.device) / T).unsqueeze(-1) if T is not None else args.steps

  with torch.no_grad():
      for image_batch in train_dataloader:
    
            #image_batch = train_data         
            image_batch = image_batch[0].to(args.device)        
            image_batch_code = my_autoencoder.encoder(image_batch)
            image_batch_recon = my_autoencoder(image_batch)
    
            num_b =  min( args.N_Selected, image_batch.shape[0] ) - 1
            image_path = []
            for i in range(num_b):                
                
                image_path.append(image_batch[i].unsqueeze(0) ) #default type is torch.float32
                for step in steps[1:T]:
                    #print(step)
                    code_sequence_i = image_batch_code[i] + step*(image_batch_code[i+1] - image_batch_code[i])
                    image_interpolated_i = my_autoencoder.decoder(torch.unsqueeze(code_sequence_i,0))
                    
                    image_path.append(image_interpolated_i.clone())   
                    
            
            image_path.append(image_batch[num_b].unsqueeze(0))      
    
            image_path = torch.cat(image_path)
            if flip_:
              image_path = 1 - image_path
            make_grid_show( image_path, pad_value= args.pad_value )
            return None


def make_grid_show(imgs,pad_value=1):
  #torchvision.utils.make_grid input: 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.
  if imgs.ndim == 3:
      imgs = torch.unsqueeze(imgs,1)
  img = make_grid(imgs,10,2,pad_value = pad_value)
  npimg = img.detach().cpu().numpy()
  plt.figure(figsize = (30,30) )
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  #plt.imshow(npimg )
  plt.axis( 'off' )
  plt.show()

def visualize_momentum(args, T=None):
  print('Warning: This is for size 32*32')
  if T == None:
    T = args.T

  if args.boundary == 'periodic':
    for t in range(T):
      X, Y = np.meshgrid(np.arange(0, 32), np.arange(0, 32))
      DX = args.pathvariables['momentum'][t,:1024].reshape(32,32)[:32,:]
      DY = args.pathvariables['momentum'][t,1024:].reshape(32,32)[:,:32]

      plt.quiver(X, Y, DY, DX)#, scale=1, scale_units='inches')

      plt.show()

  if args.boundary == 'dirichlet':
    for t in range(T):
        X, Y = np.meshgrid(np.arange(0, 32), np.arange(0, 32))
        DX = args.pathvariables['momentum'][t,:1056].reshape(33,32)[:32,:]
        DY = args.pathvariables['momentum'][t,1056:].reshape(32,33)[:,:32]

        plt.quiver(X, Y, DY, DX)#, scale=1, scale_units='inches')

        plt.show()

  if args.boundary == 'neumann':
    for t in range(T):
      X, Y = np.meshgrid(np.arange(0, 31), np.arange(0, 31))
      DX = args.pathvariables['momentum'][t,:992].reshape(31,32)[:,:31]
      DY = args.pathvariables['momentum'][t,992:].reshape(32,31)[:31,:]

      plt.quiver(X, Y, DY, DX)#, scale=1, scale_units='inches')

      plt.show()

      





def imshowpng(imgname = 'cir1.png', m=32, n=32):
  img = Image.open(imgname).convert('L').resize((m,n))
  img1 = 1-np.array(img)/255
  plt.imshow(img1,'gray')
  plt.show()


###########the code below is to generate a circle on the image
''' 
import matplotlib.patches as patches

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
circ = patches.Circle((0.5, 0.5), 0.25)
ax.add_patch(circ)
ax.set_axis_off()
'''

#################parameters adjusting/ loading#######################
def load_div_mask(args):
  v_dim  = 2*m*n+m+n
  m = args.m
  n = args.n 

  #for u whose shape is (m+1)*n, it generate 2n constraints
  top_index = np.linspace(0,n-1,num=n,dtype=int)
  bottom_index = np.linspace(m*n, m*n+n-1,num=n,dtype=int)
  #for v whose shape is m*(n+1), it generate 2m constraints
  left_index = (m*n+n) + (n+1)*np.linspace(0,m-1,num=m,dtype=int)
  right_index = (m*n+n) + n+ (n+1)*np.linspace(0,m-1,num=m,dtype=int)
  boundary_index  = sum( [top_index.tolist(),bottom_index.tolist(),left_index.tolist(),right_index.tolist()],[])

  mask = torch.ones(v_dim, dtype=bool)
  mask[boundary_index] = 0
  args.mask = mask


  divname = 'spdiv' + str(m) +'.pkl'
  with open(divname, 'rb') as f:
      args.spdiv = pickle.load(f)
      args.v_dim = DIV.shape[1]

  print('spdiv loaded, spdiv and mask updated')

# Dirichlet boundary condition, 0 at the boundary of potential
def create_div_Dirichlet(args):
  m = args.m
  n = args.n
  DIV1 = sp.eye(m*n, m*n+n, k=n) - sp.eye(m*n, m*n+n, k=0)    
  DIV2_0 = sp.eye(m*n+m, m*n+m, k=1) - sp.eye(m*n+m, m*n+m, k=0)
  index_del = n + (n+1)*np.arange(m)
  DIV2 = sp.vstack([DIV2_0[i] for i in range(DIV2_0.shape[0]) if i not in index_del])
  
  DIV = sp.hstack([DIV1, DIV2])  
  mask = np.ones(DIV.shape[1], dtype=bool)

  args.mask = torch.tensor(mask)
  args.spdiv = DIV
  args.v_dim = DIV.shape[1]
  print('spdiv and mask updated')

  Flag_ = False
  if Flag_:     
      savename = 'spdiv' + str(m) +'.pkl'

      with open(savename, 'wb') as f:
          pickle.dump(spdiv, f)

      files.download(savename)
      print('spdiv file downloaded')




def finite_difference_1d(N):
    D = np.zeros((N, N))
    np.fill_diagonal(D, -1)
    np.fill_diagonal(D[:, 1:], 1)
    #D[0, -1] = 1
    D[-1, 0] = 1
    return D

def divergence_matrix_2d(N):
    D1 = finite_difference_1d(N)
    I = np.eye(N)
    
    # Gradient in x-direction
    Dx = np.kron(D1, I)
    
    # Gradient in y-direction
    Dy = np.kron(I, D1)
    
    # Combine to get the divergence operator
    D = np.hstack([Dx, Dy])

    D = np.where(D == -0, 0, D)
    
    return D

def create_div_periodicboundary(args):
  assert args.m == args.n, "fail to generate the DIV matrix since m!=n"
  
  DIV =  sp.csr_matrix(  divergence_matrix_2d(args.m)  )  
  mask = np.ones(DIV.shape[1], dtype=bool)

  args.mask = torch.tensor(mask)
  args.spdiv = DIV
  args.v_dim = DIV.shape[1]
  print('spdiv and mask updated')

  Flag_ = False
  if Flag_:     

      savename = 'spdiv' + str(m) +'.pkl'

      with open(savename, 'wb') as f:
          pickle.dump(spdiv, f)

      files.download(savename)
      print('spdiv file downloaded')


#set 0 momentum as 0 at the boundary
def create_div_Neumann(args):
  m = args.m
  n = args.n
  DIV1 = sp.eye(m*n, m*n+n, k=n) - sp.eye(m*n, m*n+n, k=0)    
  DIV2_0 = sp.eye(m*n+m, m*n+m, k=1) - sp.eye(m*n+m, m*n+m, k=0)
  index_del = n + (n+1)*np.arange(m)
  DIV2 = sp.vstack([DIV2_0[i] for i in range(DIV2_0.shape[0]) if i not in index_del])
  
  DIV = sp.hstack([DIV1, DIV2])

  # For u whose shape is (m+1)*n, it generates 2n constraints
  top_index = np.linspace(0, n-1, num=n, dtype=int)
  bottom_index = np.linspace(m*n, m*n+n-1, num=n, dtype=int)
  
  # For v whose shape is m*(n+1), it generates 2m constraints
  left_index = (m*n+n) + (n+1)*np.linspace(0, m-1, num=m, dtype=int)
  right_index = (m*n+n) + n + (n+1)*np.linspace(0, m-1, num=m, dtype=int)
  
  boundary_index = sum([top_index.tolist(), bottom_index.tolist(), left_index.tolist(), right_index.tolist()], [])
  
  mask = np.ones(DIV.shape[1], dtype=bool)
  mask[boundary_index] = 0
  indices = np.where(mask)[0]
  spdiv = DIV[:, indices]

  args.mask = torch.tensor(mask)
  args.spdiv = spdiv  
  args.v_dim = DIV.shape[1]
  print('spdiv and mask updated')

  Flag_ = False
  if Flag_:     

      savename = 'spdiv' + str(m) +'.pkl'

      with open(savename, 'wb') as f:
          pickle.dump(spdiv, f)

      files.download(savename)
      print('spdiv file downloaded')


def expand_div(args):
  
  args.spdiv  = sp.hstack([args.spdiv, -sp.eye( args.spdiv.shape[0] )])
  print('spdiv expanded')


def import_obstacle_periodicboundary(args):
    
    if args.obstacleoption == 'default':
        # No obstacle setting
        obstacle = torch.zeros((args.m, args.n), dtype=torch.bool)
        args.obstacle = obstacle.to(dtype=torch.bool) 
        print('args.obstacle has been updated to the no obstacle setting')
    elif args.obstacleoption == 'central_line': 
        # Create a default obstacle
        obstacle = torch.zeros((args.m, args.n), dtype=torch.bool)
        obstacle[10:50, 30] = 1 #for 64*64 setting
        args.obstacle = obstacle.to(dtype=torch.bool) 
        print('args.obstacle has been updated to the default setting')
    elif args.obstacleoption == 'Labyrinthe64': 
        # Load from existing file
        obstacle = torch.load('obstacle64_new.pt')
        args.obstacle = obstacle.to(dtype=torch.bool) 
        print('args.obstacle has been updated to the Labyrinthe64 setting')
    elif args.obstacleoption == 'Labyrinthe48': 
        # Load from existing file
        obstacle = torch.load('Labyrinthe48.pt')
        args.obstacle = obstacle.to(dtype=torch.bool) 
        print('args.obstacle has been updated to the Labyrinthe48 setting')
    else:
        print('Fail to load due to wrong obstacle option')

    

    # Generate the mask with obstacle
    indices = [-1] + list(range(0, args.m-1)) #need to redefine it when m!=n
    image_j2 = 0.5*obstacle[:,indices] + 0.5*obstacle
    image_j1 = 0.5*obstacle[indices] + 0.5*obstacle
    temp = torch.hstack((torch.flatten(image_j1), torch.flatten(image_j2)))
    obstacle_mask = ~temp.to(dtype=torch.bool) 

    # Eliminate rows information of div using sparse matrix operation
    DIV = args.spdiv
    DIV = DIV[~args.obstacle.flatten()]  # Eliminate rows of onstacle

    # Eliminate columns of div 
    DIV = DIV[:, obstacle_mask]
    args.spdiv = DIV
    args.v_dim = DIV.shape[1]
    print('spdiv has been updated with obstacle')

    # Update mask
    args.mask = args.mask & obstacle_mask
    print('mask has been updated with obstacle')


  
def import_obstacle(args):
    
    if args.obstacleoption == 'default':
        # No obstacle setting
        obstacle = torch.zeros((args.m, args.n), dtype=torch.bool)
        args.obstacle = obstacle.to(dtype=torch.bool) 
        print('args.obstacle has been updated to the no obstacle setting')
    elif args.obstacleoption == 'central_line': 
        # Create a default obstacle
        obstacle = torch.zeros((args.m, args.n), dtype=torch.bool)
        obstacle[10:50, 30] = 1 #for 64*64 setting
        args.obstacle = obstacle.to(dtype=torch.bool) 
        print('args.obstacle has been updated to the default setting')
    elif args.obstacleoption == 'Labyrinthe64': 
        # Load from existing file
        obstacle = torch.load('obstacle64_new.pt')
        args.obstacle = obstacle.to(dtype=torch.bool) 
        print('args.obstacle has been updated to the Labyrinthe64 setting')
    elif args.obstacleoption == 'Labyrinthe48': 
        # Load from existing file
        obstacle = torch.load('Labyrinthe48.pt')
        args.obstacle = obstacle.to(dtype=torch.bool) 
        print('args.obstacle has been updated to the Labyrinthe48 setting')
    else:
        print('Fail to load due to wrong obstacle option')

    

    # Generate the mask with obstacle
    zero_column = torch.zeros(args.m, 1, dtype=torch.bool) 
    zero_row = torch.zeros(1, args.n, dtype=torch.bool)
    image_j2 = 0.5 * torch.cat((obstacle, zero_column), dim=1) + 0.5 * torch.cat((zero_column, obstacle), dim=1) #dtype is torch.float32 due to arithmetic operation
    image_j1 = 0.5 * torch.cat((obstacle, zero_row), dim=0) + 0.5 * torch.cat((zero_row, obstacle), dim=0) 
    temp = torch.hstack((torch.flatten(image_j1), torch.flatten(image_j2)))
    obstacle_mask = ~temp.to(dtype=torch.bool) 

    # Eliminate rows information of div using sparse matrix operation
    DIV = args.spdiv
    DIV = DIV[~args.obstacle.flatten()]  # Eliminate rows of onstacle

    # Eliminate columns of div 
    DIV = DIV[:, obstacle_mask[args.mask]]
    args.spdiv = DIV
    args.v_dim = DIV.shape[1]
    print('spdiv has been updated with obstacle')

    # Update mask
    args.mask = args.mask & obstacle_mask
    print('mask has been updated with obstacle')
    

#############load data################################# 

def loadMNIST(args, binary_=True):
    
    # Define the custom threshold function
    def threshold(x):
        return x > 0.5
    
    # Define the transforms to apply
    transform = transforms.Compose([
          transforms.Pad( (2,2,2,2)  ),       #transforms.Resize(32)
          transforms.ToTensor(),
          ])

    if binary_:
      transform = transforms.Compose([
          transforms.Pad( (2,2,2,2)  ),  #transforms.Resize(32),  # Resize to 32x32
          transforms.ToTensor(),  # Convert to tensor
          transforms.Lambda(threshold),  # Apply threshold to create binary image
          transforms.Lambda(lambda x: x.float())  # Convert binary image to float
      ])
    
      
    # Load the MNIST dataset with the defined transforms
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=args.shuffle)
    
    return train_dataloader,test_dataloader

def loadlistpt(args,filename):

  X = torch.load( filename  )
  
  X = torch.stack(X )
  X = torch.unsqueeze(X,1).float().to(args.device) #for training, the tensor should be image_channel*image_m*image_n 

  image_number,image_channel, image_m,image_n = X.shape
  print('the shape of stacked train_data  is ', image_number,image_channel, image_m,image_n)

  print('image values are between', X.min().item(), X.max().item())

  args.pad_value = X.max().item()

  #the code below returns a list object during iteration
  train_dataset = torch.utils.data.TensorDataset(X)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

  return train_dataloader
  
def load_dataset(args,filename_train='train_dataset.pth',filename_test='test_dataset.pth'):

  #the code below returns a list object during iteration
  train_dataset = torch.load( filename_train  )
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
  
  #the code below returns a list object during iteration
  test_dataset = torch.load( filename_test  )
  test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

  return train_dataloader, test_dataloader


def loadimages(args, m,n, imgs=['cir1.png','cir2.png'],flip_=True,binary_=False, gray_=False,rgb_=False):

  print('current tol is', args.tol)
  X = []
  l1_norms = []
  for imgname in imgs:
    img = Image.open(imgname).resize((n, m))
    if gray_ or binary_:
      img = img.convert('L')
      
    if rgb_:
      img = img.convert('RGB')
      args.channel = 3

    if np.max(img) > 1:
      img = np.array(img) / 255.0
    else:
      img = np.array(img)
    assert np.all((img >= 0) & (img <= 1)), "Some values in the image array are outside the range [0, 1]"

    img = torch.from_numpy(img)
    X.append( img  )
    l1_norms.append(img.sum())

  #output mass X 
  print('l1_norms in the training data:', l1_norms)

  X = torch.stack(X )
  if rgb_:
    X = X.permute(0,3,1,2).float().to(args.device)
  else:
    X = torch.unsqueeze(X,1).float().to(args.device) #for training, the tensor should be image_channel*image_m*image_n

    if binary_:
      #do a threholding here to make it binary
      threshold_ = 0.5 #this value should be
      background = (X< threshold_)
      X[ background ]   = 0.0
      X[  ~background ] = 1.0
  print('X.shape:', X.shape)

  if flip_:
    X = 1 - X
    print( "Note that the white and black values of images are flipped" ) 

  image_number,image_channel, image_m,image_n = X.shape
  print('the shape of stacked train_data  is ', image_number,image_channel, image_m,image_n)

  print('image values are between', X.min().item(), X.max().item())

  args.pad_value = X.max().item()

  #the code below returns a list object during iteration
  train_dataset = torch.utils.data.TensorDataset(X)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

  return train_dataloader

