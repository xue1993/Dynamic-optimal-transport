import time
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


class PathEnergy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, partial_f, args, result_dict):
    #DIV is sparse matrix , partial_f is a batched tensor vector, weight is also batched tensor; each batch represent variables at a time
        
        num_T = weight.shape[0]
        dim_momentum = args.v_dim
        dim_source = args.spdiv.shape[1] - dim_momentum
        DIV = args.spdiv
        ctx.DIV_T = DIV.transpose()
        ctx.device = weight.device
        dtype_ = np.double # single # 
        damp_matrix = sp.diags([1e-10]*args.spdiv.shape[0], dtype=dtype_) #due to the large condtion number, adding a damp_matrx can stablize the computation
               

        result = torch.zeros_like( partial_f )
        momentum = np.zeros((num_T, dim_momentum))
        source  = np.zeros( (num_T, dim_source) )
        
        #solve the lienar systems one by one
        A = []
        for i in range(num_T ):
          DwD = DIV @ sp.diags(weight[i].cpu().numpy()) @ ctx.DIV_T  +  damp_matrix  #DwD is in csr format
          A.append(DwD.copy())        
        A = sp.block_diag(A,format='csr').astype(dtype_)
        
        b = partial_f.flatten().cpu().numpy().astype(dtype_)
        t0 = time.time()
        temp = spsolve( A ,  b)
        print( 'innner spsolve time used: ', time.time() - t0)

        temp = temp.reshape(num_T, -1 )

        for i in range( num_T):
          
          result[i] = torch.tensor(temp[i])       
          

          if args.regu_verbosity: 

            temp_i =  sp.diags(weight[i].cpu().numpy()) @ ctx.DIV_T @ result[i].cpu().numpy()
      
            momentum[i] =  temp_i[:dim_momentum]
            source[i] = temp_i[dim_momentum:]
            
            
            if np.isnan(temp).any():
                print('WARNING: spsolve failed', np.linalg.matrix_rank(DwD.toarray()), b.sum(), weight.min().item(), weight.max().item()  ) 
            
            DwD = DIV @ sp.diags(weight[i].cpu().numpy()) @ ctx.DIV_T  +  damp_matrix
            residual_norm = np.linalg.norm( DwD.astype(dtype_) @ result[i].cpu().numpy().astype(dtype_) -  partial_f[i].cpu().numpy().astype(dtype_) )
            if residual_norm>1:
                print( 'WARNING: large spsolve residual', residual_norm ) 
      
        #save for backward
        ctx.save_for_backward( result )
        result_dict[ 'result' ] = result.detach().cpu().numpy() #then we can access the 'result' variable outside of the function. e.g, we can visualize the momentum/source term
        result_dict[ 'momentum' ] =  momentum
        result_dict[ 'source' ] =  source

        if args.regu_path_verbosity:
            path_energy = []
            momentum_energy = [ ]
            source_energy = [ ]
            for i in range( num_T):
                path_energy.append( (result[i] * partial_f[i]  ).sum().item()    )
                momentum_energy.append( momentum[i] @ sp.diags( 1 /  weight[i,:dim_momentum].cpu().numpy()) @ momentum[i] )
                source_energy.append( source[i] @  sp.diags(  1 / weight[i,dim_momentum:].cpu().numpy()) @ source[i]  )
            print('path energy:', path_energy )    
            print('momentum energy:', momentum_energy )  
            print('source energy:', source_energy )  
            

        #return the energy
        return  (result * partial_f  ).sum() 

    @staticmethod
    def backward(ctx, grad_output):
        # recover the tensor
        x = ctx.saved_tensors[0]
        grad_weight = []
        for i in range(x.shape[0]):
          grad_weight.append(torch.tensor(ctx.DIV_T @ x[i].cpu().numpy())  **2)

        return    - grad_output* torch.stack(grad_weight) .to(ctx.device),  grad_output * 2 * x,  None, None