import math
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

# JIT
from torch.utils.cpp_extension import load

dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])


def get_grid_mask(points_all, pc_range):
    masks = []
    for batch in range(points_all.shape[0]):
        points = points_all[batch].T
        mask1 = torch.logical_and(pc_range[0] <= points[0], points[0] <= pc_range[3])
        mask2 = torch.logical_and(pc_range[1] <= points[1], points[1] <= pc_range[4])
        mask3 = torch.logical_and(pc_range[2] <= points[2], points[2] <= pc_range[5])
        mask = mask1 & mask2 & mask3
        masks.append(mask)

    # print("shape of mask being returned", mask.shape)
    return torch.stack(masks)


def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
    )


def deconv3x3(in_channels, out_channels, stride):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        output_padding=1,
        bias=False,
    )


def maxpool2x2(stride):
    return nn.MaxPool2d(kernel_size=2, stride=stride, padding=0)


def relu(inplace=True):
    return nn.ReLU(inplace=inplace)


def bn(num_features):
    return nn.BatchNorm2d(num_features=num_features)


class ConvBlock(nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, max_pool=False):
        super(ConvBlock, self).__init__()

        layers = []
        for i in range(num_layer):
            _in_channels = in_channels if i == 0 else out_channels
            layers.append(conv3x3(_in_channels, out_channels))
            layers.append(bn(out_channels))
            layers.append(relu())

        if max_pool:
            layers.append(maxpool2x2(stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = num_filters[4]

        # Block 1-4
        _in_channels = self.in_channels
        self.block1 = ConvBlock(
            num_layers[0], _in_channels, num_filters[0], max_pool=True
        )
        self.block2 = ConvBlock(
            num_layers[1], num_filters[0], num_filters[1], max_pool=True
        )
        self.block3 = ConvBlock(
            num_layers[2], num_filters[1], num_filters[2], max_pool=True
        )
        self.block4 = ConvBlock(num_layers[3], num_filters[2], num_filters[3])

        # Block 5
        _in_channels = sum(num_filters[0:4])
        self.block5 = ConvBlock(num_layers[4], _in_channels, num_filters[4])

    def forward(self, x):
        N, C, H, W = x.shape

        # the first 4 blocks
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)

        # upsample and concat
        _H, _W = H // 4, W // 4
        c1_interp = F.interpolate(
            c1, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c2_interp = F.interpolate(
            c2, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c3_interp = F.interpolate(
            c3, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c4_interp = F.interpolate(
            c4, size=(_H, _W), mode="bilinear", align_corners=True
        )

        #
        c4_aggr = torch.cat((c1_interp, c2_interp, c3_interp, c4_interp), dim=1)
        c5 = self.block5(c4_aggr)

        return c5


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            deconv3x3(in_channels, 128, stride=2),
            bn(128),
            relu(),
            conv3x3(128, 128),
            bn(128),
            relu(),
            deconv3x3(128, 64, stride=2),
            bn(64),
            relu(),
            conv3x3(64, 64),
            bn(64),
            relu(),
            conv3x3(64, out_channels, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class OccupancyForecastingNetwork(nn.Module):
    def __init__(self, model_type, loss_type, n_input, n_output, pc_range, voxel_size):

        super(OccupancyForecastingNetwork, self).__init__()

        self.model_type = model_type
        assert self.model_type in ["static", "dynamic"]

        self.loss_type = loss_type.lower()
        assert self.loss_type in ["l1", "l2", "absrel"]

        self.n_input = n_input
        self.n_output = n_output

        self.n_height = int((pc_range[5] - pc_range[2]) / voxel_size)
        self.n_length = int((pc_range[4] - pc_range[1]) / voxel_size)
        self.n_width = int((pc_range[3] - pc_range[0]) / voxel_size)

        self.input_grid = [self.n_input, self.n_height, self.n_length, self.n_width]
        print("input grid:", self.input_grid)

        self.output_grid = [self.n_output, self.n_height, self.n_length, self.n_width]
        print("output grid:", self.output_grid)

        self.pc_range = pc_range
        self.voxel_size = voxel_size

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        )
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size] * 3)[None, None, :], requires_grad=False
        )

        _in_channels = self.n_input * self.n_height
        self.encoder = Encoder(_in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256])

        # NOTE: initialize the linear predictor (no bias) over history
        if self.model_type == "static":
            _out_channels = self.n_height
        elif self.model_type == "dynamic":
            _out_channels = self.n_output * self.n_height

        self.linear = torch.nn.Conv2d(
            _in_channels, _out_channels, (3, 3), stride=1, padding=1, bias=True
        )

        #
        self.decoder = Decoder(self.encoder.out_channels, _out_channels)

    def set_threshs(self, threshs):
        self.threshs = torch.nn.parameter.Parameter(
            torch.Tensor(threshs), requires_grad=False
        )

    def forward(
        self,
        input_points_orig,
        input_tindex,
        output_origin_orig,
        output_points_orig,
        output_tindex,
        output_labels=None,
        loss=None,
        mode="training",
        eval_within_grid=False,
        eval_outside_grid=False
    ):

        if loss == None:
            loss = self.loss_type

        if eval_within_grid:
            inner_grid_mask = get_grid_mask(output_points_orig, self.pc_range)
        if eval_outside_grid:
            outer_grid_mask = ~ get_grid_mask(output_points_orig, self.pc_range)

        # preprocess input/output points
        input_points = ((input_points_orig - self.offset) / self.scaler).float()
        output_origin = ((output_origin_orig - self.offset) / self.scaler).float()
        output_points = ((output_points_orig - self.offset) / self.scaler).float()

        # -1: freespace, 0: unknown, 1: occupied
        # N x T1 x H x L x W
        input_occupancy = dvr.init(input_points, input_tindex, self.input_grid)

        # double check
        N, T_in, H, L, W = input_occupancy.shape
        assert T_in == self.n_input and H == self.n_height

        _input = input_occupancy.reshape(N, -1, L, W)

        # w/ skip connection
        _output = self.linear(_input) + self.decoder(self.encoder(_input))

        #
        output = _output.reshape(N, -1, H, L, W)

        if self.model_type == 'static' and self.n_output > 1:
            output = output.repeat(1, self.n_output, 1, 1, 1)

        ret_dict = {}

        if mode == "training":
            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True)

                if sigma.requires_grad:
                    pred_dist, gt_dist, grad_sigma = dvr.render(
                        sigma,
                        output_origin,
                        output_points,
                        output_tindex,
                        loss
                    )
                    # take care of nans and infs if any
                    invalid = torch.isnan(grad_sigma)
                    grad_sigma[invalid] = 0.0
                    invalid = torch.isnan(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0
                    invalid = torch.isinf(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0
                    sigma.backward(grad_sigma)
                else:
                    pred_dist, gt_dist = dvr.render_forward(
                        sigma,
                        output_origin,
                        output_points,
                        output_tindex,
                        self.output_grid,
                        "train"
                    )
                    # take care of nans if any
                    invalid = torch.isnan(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0

                pred_dist *= self.voxel_size
                gt_dist *= self.voxel_size

                # compute training losses
                valid = gt_dist >= 0
                count = valid.sum()
                l1_loss = torch.abs(gt_dist - pred_dist)
                l2_loss = ((gt_dist - pred_dist) ** 2) / 2
                absrel_loss = torch.abs(gt_dist - pred_dist) / gt_dist

                # record training losses
                if count == 0:
                    count = 1
                ret_dict["l1_loss"] = l1_loss[valid].sum() / count
                ret_dict["l2_loss"] = l2_loss[valid].sum() / count
                ret_dict["absrel_loss"] = absrel_loss[valid].sum() / count

            else:
                raise RuntimeError(f"Unknown loss type: {loss}")

        elif mode in ["testing", "plotting"]:

            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True)
                pred_dist, gt_dist = dvr.render_forward(
                    sigma, output_origin, output_points, output_tindex, self.output_grid, "test")
                pog = 1 - torch.exp(-sigma)

                pred_dist = pred_dist.detach()
                gt_dist = gt_dist.detach()

            #
            pred_dist *= self.voxel_size
            gt_dist *= self.voxel_size

            if mode == "testing":
                # L1 distance and friends
                mask = gt_dist > 0
                if eval_within_grid:
                    mask = torch.logical_and(mask, inner_grid_mask)
                if eval_outside_grid:
                    mask = torch.logical_and(mask, outer_grid_mask)
                count = mask.sum()
                l1_loss = torch.abs(gt_dist - pred_dist)
                l2_loss = ((gt_dist - pred_dist) ** 2) / 2
                absrel_loss = torch.abs(gt_dist - pred_dist) / gt_dist

                ret_dict["l1_loss"] = l1_loss[mask].sum() / count
                ret_dict["l2_loss"] = l2_loss[mask].sum() / count
                ret_dict["absrel_loss"] = absrel_loss[mask].sum() / count

                ret_dict["gt_dist"] = gt_dist
                ret_dict["pred_dist"] = pred_dist
                ret_dict['pog'] = pog.detach()
                ret_dict["sigma"] = sigma.detach()

            if mode == "plotting":
                ret_dict["gt_dist"] = gt_dist
                ret_dict["pred_dist"] = pred_dist
                ret_dict["pog"] = pog

        elif mode == "dumping":
            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True)
                pog = 1 - torch.exp(-sigma)

            pog_max, _ = pog.max(dim=1)
            ret_dict["pog_max"] = pog_max

        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        return ret_dict
