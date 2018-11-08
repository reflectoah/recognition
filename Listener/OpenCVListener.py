import cv2
import numpy as np
import roypy
import matplotlib.pyplot as plt

class OpenCVListener(roypy.IDepthDataListener):

    def __init__(self, z_queue, gray_queue, undistortImage=False):
        super(OpenCVListener, self).__init__()
        self.z_queue = z_queue
        self.gray_queue = gray_queue
        self.undistortImage = undistortImage
        self.cameraMatrix = None
        self.distortionCoefficients = None

    def onNewData(self, data):
        # this callback function will be called for every new depth frame

        # create two images which will be filled afterwards
        # each image containing one 32Bit channel
        zImage = np.zeros((data.height, data.width), dtype=np.float32)
        grayImage = np.zeros((data.height, data.width), dtype=np.float32)

        points = data.points()
        k = 0
        for y in np.arange(data.height):
            for x in np.arange(data.width):
                point = points[k]
                k += 1
                if point.depthConfidence > 0:
                    # if the point is valid, map the pixel from 3D world coordinates to a 2D plane (this will distort the image)
                    zImage[y][x] = self.adjustZValue(point.z)
                    grayImage[y][x] = self.adjustGrayValue(point.grayValue)

        # create images to store the 8Bit version (some OpenCV functions may only work on 8Bit images)
        # convert images to the 8Bit version
        # This sample uses a fixed scaling of the values to (0,255) to avoid flickering
        # You can also replace this with an automatic scaling by using normalize(zImage, zImage8, 0, 255, NORM_MINMAX, CV_8UC1)
        zImage8 = zImage.astype(np.uint8)
        grayImage8 = grayImage.astype(np.uint8)

        if self.undistortImage:
            raise NotImplementedError

        # scale and display the depth image
        scaledZImage = cv2.resize(zImage8, (data.height * 4, data.width * 4))

        # cv2.startWindowThread()
        # cv2.imshow("Depth", scaledZImage)
        self.z_queue.put(scaledZImage)

        if self.undistortImage:
            grayImage8 = cv2.undistort(grayImage8, self.cameraMatrix, self.distortionCoefficients)

        # scale and display the gray image
        scaledGrayImage = cv2.resize(grayImage8, (data.height * 4, data.width * 4))
        # cv2.imshow("Gray", scaledGrayImage)
        self.gray_queue.put(scaledGrayImage)

    def setLensParameters(self, lensParameters):
        # construct the camera matrix
        # ( fx  0  cx)
        # ( 0   fy cy)
        # ( 0   0  1 )
        self.cameraMatrix = [
            [
                lensParameters.focalLenght.first, 0, lensParameters.principalPoint.first
            ],
            [
                0, lensParameters.focalLength.second, lensParameters.principalPoint.second,
            ],
            [
                0, 0, 1
            ]
        ]

        # construct the distortion coefficients
        # k1 k2 p1 p2 k3
        self.distortionCoefficients = [
            lensParameters.distortionRadial[0],
            lensParameters.distortionRadial[1],
            lensParameters.distortionTangential.first,
            lensParameters.distortionTangential.second,
            lensParameters.distortionRadial[2]
        ]

    def toggleUndistort(self):
        self.undistortImage = not self.undistortImage

    def adjustZValue(self, zValue):
        """
        Adjust z value to fit fixed scaling, here max dist is 2.5m
        The max dist here is used as an example and can be modified
        :param zValue:
        :return:
        """
        clampedDist = min(2.5, zValue)
        return clampedDist / 2.5 * 255.

    def adjustGrayValue(self, grayValue):
        """
        Adjust gray value to fit fixed scaling, here max value is 180
        The max value here is used as an example and can be modified
        :param grayValue:
        :return:
        """
        clampedVal = min(180., grayValue)
        return clampedVal / 180. * 255.

    def paint(self, data):
        """
        Called in the main thread, with data containing one of the items that was added to the queue in onNewData
        :param data:
        :return:
        """

        # create a figure and show the raw data
        plt.figure(1)
        plt.imshow(data)

        plt.show(block=False)
        plt.draw()

        # this pause is needed to ensure the drawing for some backends
        plt.pause(0.001)
