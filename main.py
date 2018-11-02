import queue
import time

import numpy as np
import matplotlib.pyplot as plt

import cv2
import pyautogui

from utils.roypy_platform_utils import PlatformHelper
from utils.roypy_sample_utils import *
from utils.sample_camera_info import print_camera_info

TOP_LEFT_X = -0.196
TOP_LEFT_Y = 0.047
BOTTOM_RIGHT_X = 0.195
BOTTOM_RIGHT_Y = 0.746


class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.queue = q

    def onNewData(self, data):
        zValues = []
        for i in range(data.getNumPoints()):
            zValues.append(data.getZ(i))
        zArray = np.asarray(zValues)
        p = zArray.reshape(-1, data.width)
        self.queue.put(p)

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


class ReflectoahListener(roypy.IDepthDataListener):
    def __init__(self):
        super(ReflectoahListener, self).__init__()

    def onNewData(self, data):
        print("got new data")

        try:
            points = data.points()
            """
            x_max_point, x_max = None, -np.inf
            k = 0
            for y in np.arange(data.height):
                for x in np.arange(data.width):
                    # print (y,x)
                    point = points[k]
                    # 3 conditions for clipping:
                    # 1. if z>700  => NAN
                    # 2. if y>220 or y<-220 => NAN
                    # 3. if x<-5 => NAN
                    # point = self.point_cloud[y][x]
                    k += 1

                    if point.depthConfidence > 0 and not (
                            point.x > -.005 or point.x < -.100 or point.y > .220 or point.y < -.220 or point.z > .700):
                        # find highest x value in order to find finger tip
                        if point.x > x_max:
                            x_max = point.x
                            x_max_point = point

                        #zImage[y][x] = self.adjustZValue(point.z)
                        #grayImage[y][x] = self.adjustGrayValue(point.grayValue)

            self.move_mouse(x_max_point)
        """
        except Exception as e:
            print(e)

    def move_mouse(self, point):
        try:
            print("move mouse")
            if point:
                screen_size = pyautogui.size()

                x_val = (point.y - TOP_LEFT_X) / (BOTTOM_RIGHT_X - TOP_LEFT_X) * screen_size[0]
                y_val = (point.z - TOP_LEFT_Y) / (BOTTOM_RIGHT_Y - TOP_LEFT_Y) * screen_size[1]


                #pyautogui.moveTo(x_val, y_val)
        except Exception:
            print("exc")

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


def sample_retrieve_data():
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser(usage=__doc__)
    add_camera_opener_options(parser)
    parser.add_argument("--seconds", type=int, default=15, help="duration to capture data")
    options = parser.parse_args()
    opener = CameraOpener(options)
    cam = opener.open_camera()
    cam.setUseCase("MODE_5_45FPS_500")

    print_camera_info(cam)
    print("isConnected", cam.isConnected())
    print("getFrameRate", cam.getFrameRate())

    # we will use this queue to synchronize the callback with the main thread, as drawing should happen in the main thread
    q = queue.Queue()
    l = MyListener(q)
    cam.registerDataListener(l)
    cam.startCapture()

    # create a loop that will run for a time (default 15 seconds)
    process_event_queue(q, l, options.seconds)
    cam.stopCapture()


def sample_open_cv():
    platformhelper = PlatformHelper()
    # Support a '--help' command-line option
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.parse_args()

    # The rest of this function opens the first camera found
    c = roypy.CameraManager()
    l = c.getConnectedCameraList()

    print("Number of cameras connected: ", l.size())
    if l.size() == 0:
        raise RuntimeError("No cameras connected")

    id = l[0]
    cam = c.createCamera(id)
    cam.initialize()
    # print_camera_info(cam, id)
    z_queue = queue.Queue()
    gray_queue = queue.Queue()
    # l = OpenCVListener(z_queue, gray_queue)
    l = ReflectoahListener(z_queue, gray_queue)
    # status = cam.getLensParameters(lensParameters)
    # l.setLensParameters(lensParameters)

    cam.registerDataListener(l)

    # Create two windows
    """
    cv2.startWindowThread()
    cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Gray", cv2.WINDOW_AUTOSIZE)
    """
    cam.startCapture()

    currentKey = 0
    # while currentKey!=27:
    #     # wait until a key is pressed
    #     c = cv2.waitKey(0)
    #     currentKey= chr(c&255)
    #
    #     if currentKey == 'd':
    #         l.toggleUndistort()

    while True:
        pass
    # process_event_queue(z_queue, gray_queue)
    cam.stopCapture()


def sample_mouse_move():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.parse_args()

    # The rest of this function opens the first camera found
    c = roypy.CameraManager()
    l = c.getConnectedCameraList()

    print("Number of cameras connected: ", l.size())
    if l.size() == 0:
        raise RuntimeError("No cameras connected")

    id = l[0]
    cam = c.createCamera(id)
    cam.initialize()

    l = ReflectoahListener()
    # status = cam.getLensParameters(lensParameters)
    # l.setLensParameters(lensParameters)

    cam.registerDataListener(l)
    cam.startCapture()

    while True:
        pass

    cam.stopCapture()


def process_event_queue(z_queue, gray_queue, seconds=150):
    # create a loop that will run for the given amount of time
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            # try to retrieve an item from the queue
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            zImage = z_queue.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            pass
            # break
        else:
            cv2.imshow("Depth", zImage)
            cv2.waitKey(1)

        try:
            # try to retrieve an item from the queue
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            grayImage = gray_queue.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            pass
            # break
        else:
            cv2.imshow("Gray", grayImage)
            cv2.waitKey(1)


if __name__ == '__main__':
    # sample_retrieve_data()
    # sample_open_cv()
    sample_mouse_move()
