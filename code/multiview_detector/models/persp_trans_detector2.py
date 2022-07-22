import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from multiview_detector.models.resnet import resnet18
import cv2
from multiview_detector.datasets.Wildtrack import Wildtrack
from multiview_detector.utils import projection
import DCNv2

class PerspTransDetector2(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.pronums = 5
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape

        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.worldgrid2worldcoord_mat)

        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))

        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam * self.pronums)]

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            # split = 7
            # self.base_pt1 = base[:split].to('cuda:0')
            # self.base_pt2 = base[split:].to('cuda:0')
            self.base_pt = base.to('cuda:0')
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')

        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to('cuda:0')

        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam * self.pronums + 2, 512, 3, padding=1), nn.ReLU(),
                                            DCNv2.DeformConv2d(512, 512, 2, padding=2), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')

        pass

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        imgs_result = []
        for cam in range(self.num_cam):
            img_feature = self.base_pt(imgs[:, cam].to('cuda:0'))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')

            img_res = self.img_classifier(img_feature.to('cuda:0'))

            imgs_result.append(img_res)

            for ih in range(self.pronums):
                proj_mat = self.proj_mats[ih * self.num_cam + cam].repeat([B, 1, 1]).float().to('cuda:0')
                if ih == 0:
                    world_feature = kornia.warp_perspective(img_feature.to('cuda:0'), proj_mat,
                                                            self.reducedgrid_shape)
                    world_features.append(world_feature.to('cuda:0'))

                else:
                    world_feature = kornia.warp_perspective(img_feature.to('cuda:0'), proj_mat,
                                                            self.reducedgrid_shape)
                    world_features.append(world_feature.to('cuda:0'))

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)

        map_result = self.map_classifier(world_features.to('cuda:0'))
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
        return map_result, imgs_result

    def get_imgcoord2worldgrid_matrices(self, worldgrid2worldcoord_mat):
        dataset = Wildtrack('D:/Python Program/MVDet/Data/Wildtrack')
        height = [0, 30, 60, 90, 15]
        projection_matrices = {}
        count = -1
        for iih in range(self.pronums):
            count += 1
            for cam in range(self.num_cam):
                xi = np.arange(0, 480, 40)
                yi = np.arange(0, 1440, 40)
                world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
                world_coord = dataset.get_worldcoord_from_worldgrid(world_grid)
                img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.intrinsic_matrices[cam],
                                                                      dataset.extrinsic_matrices[cam], height[iih])

                img_n = []
                world_n = []
                for j in range(img_coord.shape[1]):
                    if 0 < img_coord[0, j] < 1920 and 0 < img_coord[1, j] < 1080:
                        img_n.append((img_coord[0, j], img_coord[1, j]))
                        world_n.append((world_coord[0, j], world_coord[1, j]))

                Homo, mask = cv2.findHomography(np.float32(world_n), np.float32(img_n), cv2.RANSAC, 5)
                worldcoord2imgcoord_mat = np.float32(Homo)

                worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat

                imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)

                permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
                projection_matrices[count * self.num_cam + cam] = permutation_mat @ imgcoord2worldgrid_mat
                pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret
