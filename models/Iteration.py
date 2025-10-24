import torch
import torch.nn

import src.utils.Utils as Utils

from models.CircularConv2d import CircularConv2d

class Iteration(torch.nn.Module):

    # Static attribute
    # <=> if attributs change, all instance change
    # <=> attribute shared between all "Iteration" object 
    # https://docs.python.org/3/tutorial/classes.html#class-and-instance-variables
    # This attribute are not learnable
    #d_x: torch.Tensor # Shared
    #d_y: torch.Tensor # Shared
    #b_x: torch.Tensor # Shared
    #b_y: torch.Tensor # Shared
    # f: torch.Tensor # shared

    def __init__(self, 
        nb_intermediate_channels: int, 
        kernel_size: tuple,
        alpha: float,
        beta0: float,
        beta1: float,
        sigma: float,
        alpha_learnable: bool,
        beta0_learnable: bool,
        beta1_learnable: bool,
        sigma_learnable: bool,
        taylor_nb_iterations: int,
        taylor_kernel_size: tuple
    ) -> None:

        super(Iteration, self).__init__()
        # f_approx = argmin { 
        #   (alpha / 2) || g - Hf ||^{2}_{2}
        #   + (beta0 / 2) || nabla f ||^{2}_{2}
        #   + beta1 || nabla f ||_{1}}

        self.nb_intermediate_channels = nb_intermediate_channels
        self.kernel_size = kernel_size
        self.n = taylor_nb_iterations
        
        # Hyper-parameters
        self.alpha = torch.nn.Parameter(
            data=torch.tensor([alpha], dtype=torch.float),
            requires_grad=alpha_learnable
        )

        self.beta0 = torch.nn.Parameter(
            data=torch.tensor([beta0], dtype=torch.float),
            requires_grad=beta0_learnable
        )

        self.beta1 = torch.nn.Parameter(
            data=torch.tensor([beta1], dtype=torch.float),
            requires_grad=beta1_learnable
        )

        self.sigma = torch.nn.Parameter(
            data=torch.tensor([sigma], 
            dtype=torch.float), 
            requires_grad=sigma_learnable
        )

        ## H
        self.h = CircularConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=False
        )

         
    def forward(self, STg, decim_row, decim_col, d_x, d_y, b_x, b_y) -> torch.Tensor:
        # COMPUTE f approximation
        ## = (nabla_x)^{T} (d_x - b_x)
        gradT_x = Utils.dxT(d_x - b_x)
        ## = (nabla_y)^{T} (d_y - b_y)
        gradT_y = Utils.dyT(d_y - b_y)
        
        
        ## = sigma * [ (nabla_x)^{T} (d_x - b_x) + (nabla_y)^{T} (d_y - b_y) ]
        sigma_expr = self.sigma * ( gradT_x + gradT_y )

        ## = alpha * (H^{T} S^{T} g)
        alpha_expr = self.alpha * self.h.T(STg.unsqueeze(0))

        ## = [ alpha H^{T} S^{T} S H + (beta0 + sigma) laplacian ]^{-1}
        ##  * (
        ##      sigma * [ (nabla_x)^{T} (d_x - b_x) + (nabla_y)^{T} (d_y - b_y) ]
        ##      + alpha * (H^{T} S^{T} g)
        ##  )
        f = self.taylor_young_ld(
            x = (sigma_expr + alpha_expr).squeeze(0), 
            decim_row = decim_row,
            decim_col = decim_col,
            n = self.n
        )

        # Update (d_x, d_y) : Multidimensional Soft Thresholding
        dx_f = Utils.dx(f)
        dy_f = Utils.dy(f)
        d_x, d_y = Utils.multidimensional_soft(
            torch.concat(
                [ 
                    (dx_f + b_x).unsqueeze(0), 
                    (dy_f + b_y).unsqueeze(0)
                ],
                0
            ),
            self.beta1 / self.sigma
        )
      
        # Update (b_x, b_y)
        b_x += (dx_f - d_x)
        b_y += (dy_f - d_y)

        
        return [ f, d_x, d_y, b_x, b_y ]


    def taylor_young_ld(self, x: torch.Tensor, decim_row: int, decim_col: int, n: int) -> torch.Tensor:
        ld = x
        for k in range(1, n+1):
            ld = x - self.compute(ld, decim_row, decim_col)
        return ld

    def compute(self, u: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
        """Computes :
        [I - (alpha H^{T} S^{T} S H + (beta0 + sigma) laplacian)] u
        = u - [(alpha H^{T} S^{T} S H] u - [(beta0 + sigma) laplacian)] u
        """
        
        # [ alpha H^{T} S^{T} S H ] u
        out1 = self.h(u.unsqueeze(0))
        out2 = Utils.decimation(out1.squeeze(0), decim_row, decim_col)
        out3 = Utils.decimation_adjoint(out2, decim_row, decim_col)
        out4 = self.h.T(out3.unsqueeze(0))
        out5 = self.alpha * out4
        term1 = out5.squeeze(0)

        # [(beta0 + sigma) laplacian ] u
        laplacian = Utils.laplacian2D_v2(u)
        term2 = (self.beta0 + self.sigma) * laplacian
      
        # [ I - (alpha H^{T} S^{T} S H + (beta0 + sigma) laplacian) ] u
        #= u - [ (alpha H^{T} S^{T} S H ] u - [ (beta0 + sigma) laplacian) ] u
        res = u - term1 - term2

        return res