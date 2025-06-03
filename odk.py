import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import solve


class ODKConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        r: int = 2,
        num_matrices: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.kwargs = kwargs

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = None

        self.r = r
        self.width = self.in_channels * self.kernel_size * self.kernel_size

        self.max_num_matrices = self.width // self.out_channels
        assert num_matrices <= self.max_num_matrices
        self.num_matrices = num_matrices

        assert self.out_channels % self.r == 0
        assert self.out_channels <= self.width

        self.block_size = self.out_channels // self.r
        self.upper_params = self.block_size * (self.block_size - 1) // 2

        self.block = nn.Parameter(
            torch.zeros(self.num_matrices, self.r, self.upper_params)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orth_block = self.cayley_batch()
        oft_rotation = self.block_diagonal(orth_block)  # conv x conv

        oft_rotation = oft_rotation.view(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        result = F.conv2d(
            input=x,
            weight=oft_rotation,
            bias=self.bias,
            padding=self.padding,
            stride=self.stride,
        )
        return result

    def vec_to_skew_symmetric(self, vec: torch.Tensor) -> torch.Tensor:
        batch_size = vec.shape[0]
        skew_mat = torch.zeros(
            batch_size, self.block_size, self.block_size, device=vec.device
        )
        idx = torch.triu_indices(
            self.block_size, self.block_size, offset=1
        )  # индексы над диагональю
        skew_mat[:, idx[0], idx[1]] = vec
        skew_mat[:, idx[1], idx[0]] = -vec
        return skew_mat

    def cayley_batch(self) -> torch.Tensor:
        num_matrices, r, upper_params = self.block.shape

        data_flat = self.block.view(-1, upper_params)
        skew_flat = self.vec_to_skew_symmetric(data_flat)
        id_mat = (
            torch.eye(self.block_size, device=self.block.device)
            .unsqueeze(0)
            .expand(skew_flat.shape[0], self.block_size, self.block_size)
        )

        Q_flat = solve(id_mat + skew_flat, id_mat - skew_flat, left=False)
        Q = Q_flat.view(num_matrices, r, self.block_size, self.block_size)
        return Q

    def block_diagonal(self, block: torch.Tensor) -> torch.Tensor:
        matrices = []
        num_matrices = block.shape[0]
        for i in range(num_matrices):
            blocks_i = [block[i, j] for j in range(self.r)]  # i-й набор блоков
            A_i = torch.block_diag(*blocks_i)
            matrices.append(A_i)

        result = torch.cat(matrices, dim=1)

        zeros = torch.zeros(
            self.out_channels,
            self.width - num_matrices * self.out_channels,
            device=result.device,
        )
        result = torch.cat([result, zeros], dim=1)

        return result

    def __repr__(self) -> str:
        # Формируем строку с параметрами слоя
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"bias={self.bias is not None}, "
            f"r={self.r}, "
            f"num_matrices={self.num_matrices})"
        )
