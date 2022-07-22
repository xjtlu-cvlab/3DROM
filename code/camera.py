import numpy as np
import cameraModel
import cv2
import os

class camera:
    def __init__(self, scale=None, shift=None, frameScale=None):
        self.cam = cameraModel.cameraModel()

        if scale is None:
            self.scale = 1
        else:
            self.scale = scale

        if shift is None:
            self.shift = 0
        else:
            self.shift = shift

        if frameScale is None:
            self.frameScale = 1
        else:
            self.frameScale = frameScale

    def readNextFrame(self):
        ret, self.frame = self.cap.read()

    def imageToWorld(self, x, y, Z):
        x = x * self.frameScale
        y = y * self.frameScale
        return self.cam.imageToWorld(x, y, Z)

    def imageToTop(self, x, y, Z):
        x = x * self.frameScale
        y = y * self.frameScale
        X, Y = self.cam.imageToWorld(x, y, Z)
        X = X / self.scale + self.shift
        Y = Y / self.scale + self.shift
        return X, Y

    def worldToImage(self, X, Y, Z):
        x, y = self.cam.worldToImage(X, Y, Z)
        x = x / self.frameScale
        y = y / self.frameScale
        return x, y

    def topToImage(self, X, Y, Z):
        X = (X - self.shift) * self.scale
        Y = (Y - self.shift) * self.scale
        x, y = self.cam.worldToImage(X, Y, Z)
        x = x / self.frameScale
        y = y / self.frameScale
        return x, y

    def topToWorld(self, X, Y, Z):
        return (X - self.shift) * self.scale, (Y - self.shift) * self.scale, Z

    def worldToTop(self, X, Y, Z):
        return X / self.scale + self.shift, Y / self.scale + self.shift, Z

    def cameraPositionTop(self):
        return int(self.cam.mCposx/self.scale + self.shift), int(self.cam.mCposy/self.scale + self.shift)


