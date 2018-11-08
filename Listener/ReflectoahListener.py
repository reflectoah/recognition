import logging
import time
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import pyautogui

import roypy

TOP_LEFT_X = -0.196
TOP_LEFT_Y = 0.047
BOTTOM_RIGHT_X = 0.195
BOTTOM_RIGHT_Y = 0.746

MIN_X_DIST = -.005
MAX_X_DIST = -.100
MIN_Y_DIST = .220
MAX_Y_DIST = -.220
MIN_Z_DIST = .700

CLICK_TIMEOUT = 1
CLICK_THRESHOLD = -0.007

screen_size = pyautogui.size()

logger = logging.getLogger("reflectoah")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class MouseMoveThread(Thread):
    def __init__(self, y, z):
        super(MouseMoveThread, self).__init__()
        self.y = y
        self.z = z

    def run(self):
        x_val = (self.y - TOP_LEFT_X) / (BOTTOM_RIGHT_X - TOP_LEFT_X) * screen_size[0]
        y_val = (self.z - TOP_LEFT_Y) / (BOTTOM_RIGHT_Y - TOP_LEFT_Y) * screen_size[1]

        pyautogui.moveTo(x_val, y_val, _pause=False)


class ReflectoahListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(ReflectoahListener, self).__init__()
        self.q = q
        self.last_click = time.time()

    def onNewData(self, data):
        start_time = time.time()
        logger.debug("got new data")

        # zvalues = []

        x_max_point_index, x_max = None, -np.inf
        for i in range(data.getNumPoints()):
            # 3 conditions for clipping:
            # 1. if z>700  => NAN
            # 2. if y>220 or y<-220 => NAN
            # 3. if x<-5 => NAN

            if data.getDepthConfidence(i) > 0 and not (
                    data.getX(i) > MIN_X_DIST or data.getX(i) < MAX_X_DIST or data.getY(i) > MIN_Y_DIST or data.getY(
                i) < MAX_Y_DIST or data.getZ(i) > MIN_Z_DIST):
                # find highest x value in order to find finger tip
                # zvalues.append(data.getZ(i))
                x = data.getX(i)

                # check if point is closest point to mirror
                if x > x_max:
                    x_max = x
                    x_max_point_index = i

            else:
                pass
                # zvalues.append(0)
        # zarray = np.asarray(zvalues)
        # p = zarray.reshape(-1, data.width)
        # self.q.put(p)

        logger.debug("x_max: {}\nx_max_ind: {}\n\n".format(x_max, x_max_point_index))

        if x_max_point_index:

            # check if user is clicking
            if x_max > CLICK_THRESHOLD:
                logger.info("CLICK")
                self.click()

            mmt = MouseMoveThread(data.getY(x_max_point_index),
                                  data.getZ(x_max_point_index))
            mmt.start()

        end_time = time.time()

    def click(self):
        if time.time() > self.last_click + CLICK_TIMEOUT:
            pyautogui.click()
            self.last_click = time.time()

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

    def move_mouse_by_coords(self, x, y, z):
        logger.debug("move mouse")

        x_val = (y - TOP_LEFT_X) / (BOTTOM_RIGHT_X - TOP_LEFT_X) * screen_size[0]
        y_val = (z - TOP_LEFT_Y) / (BOTTOM_RIGHT_Y - TOP_LEFT_Y) * screen_size[1]

        pyautogui.moveTo(x_val, y_val, _pause=False)

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
