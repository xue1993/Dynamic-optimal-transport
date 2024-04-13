from utils import *
from model import *
from train_OT import train

def main_OT(mass_weight_, energy_weight_, boundary_ )
  #training parameter settings and some global variables are also saved here
  #there are many many parameters which is not used for now, the important ones are commented
  class Args:
      def __init__(self):
          #training parameters
          self.num_epochs = 300
          self.batch_size = 256
          self.mass_weight = mass_weight_
          self.energy_weight = energy_weight_ #IMPORTANT!, a good energy weights balance well the energy value with data fedelity term(MSE term)
          self.learning_rate = 1e-3
          self.use_gpu = True
          self.device = torch.device("cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu")
          self.N_Selected = 8
          self.T = 19  #important, it determine how many intermidiate images we have
          self._tol = float(1e-2) #to threshold the images, 0 will cause singularity. 1e-3 works well
          self.tau  = 10000
          self.boundary = boundary_ #'periodic', 'dirichlet', 'neumann'
          self.bceloss = True
          self.imgcuroption = 'mid'
          #training dataset
          self.m = 32  #important, large image size cause slow computation of the subproblem #when choose then size of images, it'd have to be from 18,24,32,40,48 (the multiple of 8 since the NN has 4 pooling layer)
          self.n = 32  #same as m
          self.channel = 1
          self.shuffle = True
          self._pad_value = 1
          self._mass_standard = 400
          self.obstacleoption = 'default'
          self.first_and_last = None
          self.flip = True
          #global variables
          self.v_dim = 2*self.m*self.n+self.m+self.n
          self.spdiv = None #this is the sparse divergence operator, since creating it every time requires large RAM when m n are large, so it's easier to load it from local
          self.mask = None # this is due to boundary condition of divergence operator, For now, the momentum on the boundary are eliminated. In the future, we consider block some area
          self.obstacle = None
          self.steps = (torch.arange(self.T+1, device=self.device) / self.T).unsqueeze(-1)
          self.pathvariables = None
          #verbosity options
          self.regu_verbosity = True
          self.regu_path_verbosity = True
          self.imshow_gap = 150
          self.save_gap = 1001

      @property
      def mass_standard(self):
          return self._mass_standard

      @mass_standard.setter
      def mass_standard(self, value):
          print('args.mass_standard changed from ',  self._mass_standard, ' to new value', value)
          self._mass_standard = float(value)

      @property
      def tol(self):
          return self._tol

      @tol.setter
      def tol(self, value):
          print('args.tol changed from ', self._tol, ' to new value', value)
          self._tol = float(value)

      @property
      def diff_tol(self):
          return self._diff_tol

      @diff_tol.setter
      def diff_tol(self, value):
          print('args.diff_tol changed from ', self._diff_tol  ,'to new value', value)
          self._diff_tol = float(value)

      @property
      def pad_value(self):
          return self._pad_value

      @pad_value.setter
      def pad_value(self, value):
          print('args.pad_value changed from', self._pad_value ,' to new value', value)
          self._pad_value = float(value)

  args = Args()


  #%%script false --no-raise-error

  # step 1: load training data, currently only two images
  #train_dataloader = loadtwoimages(args, imgs=['cir5.png','cir6.png'])
  train_dataloader = loadimages(args,128,128, imgs=['duck2.png','heart2.png', 'redcross2.png', 'tooth2.png' ],flip_= args.flip,gray_=True)


  # step 2: create or load divergence operator and mask
  if args.boundary == 'periodic':
    create_div_periodicboundary(args)
    import_obstacle_periodicboundary(args)
  else:

    if args.boundary == 'neumann':
      create_div_Neumann(args)
    else:
      create_div_Dirichlet(args)

    import_obstacle(args)

  #step 3: create the NN
  my_autoencoder = Autoencoder_s().to(args.device)
  from torchsummary import summary
  summary( my_autoencoder, (args.channel, args.m,args.n),1)

  losses = {
        "train_loss_avg":[],
        "mseterm":[],
        "massterm":[],
        "pathenergy":[]
    }
  optimizer = torch.optim.Adam(params=my_autoencoder.parameters(), lr=args.learning_rate)


  train(args, train_dataloader, my_autoencoder,optimizer,  losses, num_epochs= args.num_epochs )


  #fig = plt.figure(10,10)
  plt.plot(losses["train_loss_avg"])
  plt.xlabel('Epochs')
  plt.ylabel('Reconstruction error')
  plt.show()

  visualize_path(args, train_dataloader, my_autoencoder,flip_= False, T=9)
