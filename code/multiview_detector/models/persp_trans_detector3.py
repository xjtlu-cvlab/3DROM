import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from multiview_detector.models.resnet import resnet18
import cv2
from multiview_detector.datasets.Terrace import Terrace
import DCNv2
import copy


class PerspTransDetector3(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.cudas = 'cuda:0'
        self.pronums = 5
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape

        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices()

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
            split = 7
            # self.base_pt1 = base[:split].to('cuda:0')
            # self.base_pt2 = base[split:].to('cuda:0')
            self.base_pt = base.to(self.cudas)
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')

        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to(self.cudas)

        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam * 5 + 2, 512, 3, padding=1), nn.ReLU(),
                                            DCNv2.DeformConv2d(512, 512, 2, padding=2), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to(self.cudas)

        pass

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        imgs_result = []
        for cam in range(self.num_cam):
            img_feature = self.base_pt(imgs[:, cam].to(self.cudas))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to(self.cudas))
            imgs_result.append(img_res)

            for ih in range(self.pronums):
                proj_mat = self.proj_mats[ih * self.num_cam + cam].repeat([B, 1, 1]).float().to(self.cudas)
                if ih == 0:
                    world_feature = kornia.warp_perspective(img_feature.to(self.cudas), proj_mat,
                                                            self.reducedgrid_shape)
                    world_features.append(world_feature.to(self.cudas))

                else:
                    world_feature = kornia.warp_perspective(img_feature.to(self.cudas), proj_mat,
                                                            self.reducedgrid_shape)
                    world_features.append(world_feature.to(self.cudas))

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to(self.cudas)], dim=1)

        map_result = self.map_classifier(world_features.to(self.cudas))
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')

        return map_result, imgs_result

    def get_imgcoord2worldgrid_matrices(self):
        import camera
        camera = [camera.camera(frameScale=2),
                  camera.camera(frameScale=2),
                  camera.camera(frameScale=2),
                  camera.camera(frameScale=2)]

        camera[0].cam.setExtrinsic(-4.8441913843e+03, 5.5109448682e+02, 4.9667438357e+03, 1.9007833770e+00,
                                   4.9730769727e-01, 1.8415452559e-01)
        camera[0].cam.setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02,
                                  2.3000000000e-02)
        camera[0].cam.setIntrinsic(20.161920, 5.720865e-04, 366.514507, 305.832552, 1)
        camera[0].cam.internalInit()

        camera[1].cam.setExtrinsic(-65.433635, 1594.811988, 2113.640844, 1.9347282363e+00, -7.0418616982e-01,
                                   -2.3783238362e-01)
        camera[1].cam.setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02,
                                  2.3000000000e-02)
        camera[1].cam.setIntrinsic(19.529144, 5.184242e-04, 360.228130, 255.166919, 1)
        camera[1].cam.internalInit()

        camera[2].cam.setExtrinsic(1.9782813424e+03, -9.4027627332e+02, 1.2397750058e+04, -1.8289537286e+00,
                                   3.7748154985e-01, 3.0218614321e+00)
        camera[2].cam.setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02,
                                  2.3000000000e-02)
        camera[2].cam.setIntrinsic(19.903218, 3.511557e-04, 355.506436, 241.205640, 1.0000000000e+00)
        camera[2].cam.internalInit()

        camera[3].cam.setExtrinsic(4.6737509054e+03, -2.5743341287e+01, 8.4155952460e+03, -1.8418460467e+00,
                                   -4.6728290805e-01, -3.0205552749e+00)
        camera[3].cam.setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02,
                                  2.3000000000e-02)
        camera[3].cam.setIntrinsic(20.047015, 4.347668e-04, 349.154019, 245.786168, 1)
        camera[3].cam.internalInit()
        dataset = Terrace('D:/MVDet/Data/Terrace')
        height = [0, 300, 600, 900, 150]
        projection_matrices = {}
        count = -1

        xx, yy = np.meshgrid(np.arange(0, 360, 2), np.arange(0, 288, 2))
        H, W = xx.shape
        image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])

        for iih in range(self.pronums):
            count += 1
            for cam in range(self.num_cam):
                imgxs = []
                worldxs = []
                image_trans = copy.deepcopy(image_coords)
                image_trans2 = copy.deepcopy(image_coords)
                for ids in range(len(image_trans)):
                    imgx = image_trans[ids][0]
                    imgy = image_trans[ids][1]
                    wdx, wdy = camera[cam].imageToWorld(imgx, imgy, height[iih])
                    if wdx > 10000000 or wdy > 10000000:
                        image_trans[ids][0] = 10000000
                        image_trans[ids][1] = 10000000
                    elif wdx < -10000000 or wdy < -10000000:
                        image_trans[ids][0] = -10000000
                        image_trans[ids][1] = -10000000
                    else:
                        image_trans[ids][0] = wdx
                        image_trans[ids][1] = wdy

                world_coords = image_trans.transpose()
                image_trans2 = image_trans2.reshape([H, W, 2])
                world_grids = dataset.get_worldgrid_from_worldcoord(world_coords).transpose().reshape(
                    [H, W, 2])
                for i in range(H):
                    for j in range(W):
                        x, y = world_grids[i, j]
                        if dataset.indexing == 'xy':
                            if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
                                imgxs.append(image_trans2[i, j])
                                worldxs.append(world_grids[i, j])
                Homo, mask = cv2.findHomography(np.float32(imgxs), np.float32(worldxs), cv2.RANSAC, 5)
                projection_matrices[count * self.num_cam + cam] = Homo
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
