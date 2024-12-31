import torch


class CrossScan_rgbt_k4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_sub: torch.Tensor, x_vi: torch.Tensor, x_ir: torch.Tensor):
        # B, C, H, W -> B, 2, C, 2 * H * W
        B, C, H, W = x_vi.shape
        ctx.shape = (B, C, H, W)
        x_fus = x_vi.new_empty((B, 4, C, 2 * H * W))
        x_fus[:, 0] = torch.concat([x_vi.flatten(2, 3), x_sub.flatten(2, 3)], dim=2)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        x_fus[:, 2] = torch.concat([x_ir.flatten(2, 3), x_sub.flatten(2, 3)], dim=2)
        x_fus[:, 3] = torch.flip(x_fus[:, 2], dims=[-1])
        return x_fus

    @staticmethod
    def backward(ctx, x_fus: torch.Tensor):
        # out: (b, 2, d, l)
        B, C, H, W = ctx.shape
        # L = 2 * H * W
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W
        x_fus_2 = x_fus[:, 2] + x_fus[:, 3].flip(dims=[-1])

        # get B, d, H*W
        return (
            ((x_fus_1[:, :, H * W : 2 * H * W] + x_fus_2[:, :, H * W : 2 * H * W]) / 2).view(B, -1, H, W),
            x_fus_1[:, :, 0 : H * W].view(B, -1, H, W),
            x_fus_2[:, :, 0 : H * W].view(B, -1, H, W),
        )


class CrossMerge_rgbt_k4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_fus: torch.Tensor):
        B, K, D, L = x_fus.shape
        # ctx.shape = (H, W)
        # ys = ys.view(B, K, D, -1)
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
        x_fus_2 = x_fus[:, 2] + x_fus[:, 3].flip(dims=[-1])
        # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
        return (
            (x_fus_1[:, :, L // 2 : L] + x_fus_2[:, :, L // 2 : L]) / 2,
            x_fus_1[:, :, 0 : L // 2],
            x_fus_2[:, :, 0 : L // 2],
        )

    @staticmethod
    def backward(ctx, x_sub: torch.Tensor, x_vi: torch.Tensor, x_ir: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        # H, W = ctx.shape
        B, C, L = x_vi.shape
        x_fus = x_vi.new_empty((B, 4, C, 2 * L))

        x_fus[:, 0] = torch.cat([x_vi, x_sub], dim=2)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        x_fus[:, 2] = torch.cat([x_ir, x_sub], dim=2)
        x_fus[:, 3] = torch.flip(x_fus[:, 2], dims=[-1])
        x_fus = x_fus.view(B, 4, C, 2 * L)
        return x_fus
