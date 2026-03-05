import torch
import torch.nn as nn
import torch.nn.functional as F


def op_A(x: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
    return F.avg_pool2d(
        x.unsqueeze(0),
        kernel_size=(decim_row, decim_col),
        stride=(decim_row, decim_col),
    ).squeeze(0)


def op_At(y: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
    return F.interpolate(
        y.unsqueeze(0),
        scale_factor=(decim_row, decim_col),
        mode="nearest",
    ).squeeze(0)


def op_AtA(x: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
    return op_At(op_A(x, decim_row, decim_col), decim_row, decim_col)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels: int = 1, n_blocks: int = 2):
        super().__init__()
        layers = []
        for _ in range(n_blocks):
            layers += [
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True),
            ]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class NeumannNet(nn.Module):
    def __init__(
        self,
        nb_iteration: int = 5,
        n_channels: int = 1,
        n_residual_blocks: int = 2,
        eta_init: float = 0.1,
    ):
        super().__init__()
        self.num_iters = nb_iteration
        self.eta = nn.Parameter(torch.tensor(eta_init))
        self.regularizers = nn.ModuleList(
            [ResidualBlock(n_channels, n_residual_blocks) for _ in range(nb_iteration)]
        )

    @classmethod
    def from_config(cls, config: dict) -> "NeumannNet":
        p = config["model"]["params"]
        return cls(
            nb_iteration=p["nb_iteration"],
            n_channels=p.get("n_channels", 1),
            n_residual_blocks=p.get("n_residual_blocks", 2),
            eta_init=p["beta0"]["initialize"],
        )

    def forward(
        self,
        y: torch.Tensor,
        decim_row: int,
        decim_col: int,
        sigma: float = 0.0,  
    ) -> torch.Tensor:
        if y.dim() == 2:
            y = y.unsqueeze(0)  

        x0 = self.eta * op_At(y, decim_row, decim_col)
        runner      = x0
        neumann_sum = runner
        for ii in range(self.num_iters):
            linear_component = runner - self.eta * op_AtA(runner, decim_row, decim_col)
            learned_component = -self.regularizers[ii](runner.unsqueeze(0)).squeeze(0)

            runner      = linear_component + learned_component
            neumann_sum = neumann_sum + runner
        return neumann_sum  

    def get_metrics(self) -> dict:
        return {"eta": [self.eta.item()]}