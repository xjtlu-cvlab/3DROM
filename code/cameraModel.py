import math

class cameraModel:
    def __init__(self):
        self.isInit = 0
        self.mName = 0

        self.mImgWidth = 0
        self.mImgHeight = 0
        self.mNcx = 0
        self.mNfx = 0
        self.mDx = 0
        self.mDy = 0
        self.mDpx = 0
        self.mDpy = 0

        self.mFocal = 0
        self.mKappa1 = 0
        self.mCx = 0
        self.mCy = 0
        self.mSx = 0

        self.mTx = 0
        self.mTy = 0
        self.mTz = 0
        self.mRx = 0
        self.mRy = 0
        self.mRz = 0

        self.mR11 = 0
        self.mR12 = 0
        self.mR13 = 0
        self.mR21 = 0
        self.mR22 = 0
        self.mR23 = 0
        self.mR31 = 0
        self.mR32 = 0
        self.mR33 = 0

        self.mCposx = 0
        self.mCposy = 0
        self.mCposz = 0

    def internalInit(self):
        sa = math.sin(self.mRx)
        ca = math.cos(self.mRx)
        sb = math.sin(self.mRy)
        cb = math.cos(self.mRy)
        sg = math.sin(self.mRz)
        cg = math.cos(self.mRz)

        self.mR11 = cb * cg
        self.mR12 = cg * sa * sb - ca * sg
        self.mR13 = sa * sg + ca * cg * sb
        self.mR21 = cb * sg
        self.mR22 = sa * sb * sg + ca * cg
        self.mR23 = ca * sb * sg - cg * sa
        self.mR31 = -sb
        self.mR32 = cb * sa
        self.mR33 = ca * cb

        self.mCposx = -(self.mTx * self.mR11 + self.mTy * self.mR21 + self.mTz * self.mR31)
        self.mCposy = -(self.mTx * self.mR12 + self.mTy * self.mR22 + self.mTz * self.mR32)
        self.mCposz = -(self.mTx * self.mR13 + self.mTy * self.mR23 + self.mTz * self.mR33)

        self.isInit = 1

    def setGeometry(self, width, height, ncx, nfx, dx, dy, dpx, dpy):
        self.mImgWidth = width
        self.mImgHeight = height
        self.mNcx = ncx
        self.mNfx = nfx
        self.mDx = dx
        self.mDy = dy
        self.mDpx = dpx
        self.mDpy = dpy

        self.isInit = 0

    def setIntrinsic(self,focal , kappa1, cx, cy, sx):
        self.mFocal = focal
        self.mKappa1 = kappa1
        self.mCx = cx
        self.mCy = cy
        self.mSx = sx

        self.isInit = 0

    def setExtrinsic(self, tx, ty, tz, rx, ry, rz):
        self.mTx = tx
        self.mTy = ty
        self.mTz = tz
        self.mRx = rx
        self.mRy = ry
        self.mRz = rz

        self.isInit = 0

    def imageToWorld(self, Xi, Yi, Zw):
        if self.isInit == 1:
            Xd = self.mDpx * (Xi - self.mCx) / self.mSx
            Yd = self.mDpy * (Yi - self.mCy)

            Xu, Yu = self.distortedToUndistortedSensorCoord(Xd, Yd)

            common_denominator = ((self.mR11 * self.mR32 - self.mR12 * self.mR31) * Yu +
                                  (self.mR22 * self.mR31 - self.mR21 * self.mR32) * Xu -
                                  self.mFocal * self.mR11 * self.mR22 + self.mFocal * self.mR12 * self.mR21)

            Xw = (((self.mR12 * self.mR33 - self.mR13 * self.mR32) * Yu +
                   (self.mR23 * self.mR32 - self.mR22 * self.mR33) * Xu -
                   self.mFocal * self.mR12 * self.mR23 + self.mFocal * self.mR13 * self.mR22) * Zw +
                  (self.mR12 * self.mTz - self.mR32 * self.mTx) * Yu +
                  (self.mR32 * self.mTy - self.mR22 * self.mTz) * Xu -
                  self.mFocal * self.mR12 * self.mTy + self.mFocal * self.mR22 * self.mTx) / common_denominator

            Yw = -(((self.mR11 * self.mR33 - self.mR13 * self.mR31) * Yu +
                    (self.mR23 * self.mR31 - self.mR21 * self.mR33) * Xu -
                    self.mFocal * self.mR11 * self.mR23 + self.mFocal * self.mR13 * self.mR21) * Zw +
                   (self.mR11 * self.mTz - self.mR31 * self.mTx) * Yu +
                   (self.mR31 * self.mTy - self.mR21 * self.mTz) * Xu -
                   self.mFocal * self.mR11 * self.mTy + self.mFocal * self.mR21 * self.mTx) / common_denominator

        return Xw, Yw

    def distortedToUndistortedSensorCoord(self, Xd, Yd):
        distortion_factor = 1 + self.mKappa1 * (Xd * Xd + Yd * Yd)
        Xu = Xd * distortion_factor
        Yu = Yd * distortion_factor
        return Xu, Yu

    def worldToImage(self, Xw, Yw, Zw):
        if self.isInit == 1:
            xc = self.mR11 * Xw + self.mR12 * Yw + self.mR13 * Zw + self.mTx
            yc = self.mR21 * Xw + self.mR22 * Yw + self.mR23 * Zw + self.mTy
            zc = self.mR31 * Xw + self.mR32 * Yw + self.mR33 * Zw + self.mTz

            Xu = self.mFocal * xc / zc
            Yu = self.mFocal * yc / zc

            Xd, Yd = self.undistortedToDistortedSensorCoord(Xu, Yu)

            Xi = Xd * self.mSx / self.mDpx + self.mCx
            Yi = Yd / self.mDpy + self.mCy
        return Xi, Yi

    def undistortedToDistortedSensorCoord(self, Xu, Yu):
        if self.isInit == 1:
            if ((Xu == 0) and (Yu == 0)) or (self.mKappa1 == 0):
                Xd = Xu
                Yd = Yu
            else:
                Ru = math.sqrt(Xu * Xu + Yu * Yu)

                c = 1.0 / self.mKappa1
                d = -c * Ru

                Q = c / 3
                R = -d / 2
                D = Q * Q * Q + R * R

                if D>=0:
                    D = math.sqrt(D)
                    if R + D > 0:
                        S = pow(R + D, 1.0 / 3.0)
                    else:
                        S = -pow(-R - D, 1.0 / 3.0)

                    if R - D > 0:
                        T = pow(R - D, 1.0 / 3.0)
                    else:
                        T = -pow(D - R, 1.0 / 3.0)

                    Rd = S + T
                    if Rd<0:
                        Rd = math.sqrt(-1.0 / (3 * self.mKappa1))
                else:
                    D = math.sqrt(-D)
                    S = pow(math.sqrt(R * R + D * D), 1.0 / 3.0)
                    T = math.atan2(D, R) / 3
                    sinT = math.sin(T)
                    cosT = math.cos(T)

                    Rd = -S * cosT + math.sqrt(3.0) * S * sinT

                lambda1 = Rd / Ru

                Xd = Xu * lambda1
                Yd = Yu * lambda1
        return Xd, Yd








