import os
import numpy as np
import re
from torchvision.datasets import VisionDataset
import cameraModel

class Terrace(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # Terrace has xy-indexing: H*W=440*300, thus x is \in [0,300), y \in [0,440)
        # Terrace has consistent unit: milimeter (mm) for calibration & pos annotation
        self.__name__ = 'Terrace'
        self.img_shape, self.worldgrid_shape = [288, 360], [440, 300]
        self.num_cam, self.num_frame = 4, 4900
        self.indexing = 'xy'
        self.cam = cameraModel.cameraModel()
        # meter to millimeter
        self.unit = 1000

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = (pos % 30) * 10
        grid_y = (pos // 30) * 10
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return (grid_x / 10) + (int(grid_y / 10 + 0.5) * 30)

    def get_worldgrid_from_worldcoord(self, world_coord):
        coord_x, coord_y = world_coord
        grid_x = ((coord_x + 500) / 250 - 0.5) * 10
        grid_y = ((coord_y + 1500) / 250 - 0.5) * 10
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        coord_x = (grid_x / 10 + 0.5) * 250 - 500
        coord_y = (grid_y / 10 + 0.5) * 250 - 1500
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 360 - 1), min(bottom, 288 - 1)]
        return bbox_by_pos_cam
