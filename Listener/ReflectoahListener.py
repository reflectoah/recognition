import json
import logging
import time
from threading import Thread
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
        #x_val = (self.y - TOP_LEFT_X) / (BOTTOM_RIGHT_X - TOP_LEFT_X) * screen_size[0]
        #y_val = (self.z - TOP_LEFT_Y) / (BOTTOM_RIGHT_Y - TOP_LEFT_Y) * screen_size[1]

        x_val = (self.y - TOP_LEFT_X) / (BOTTOM_RIGHT_X - TOP_LEFT_X) * 2160
        y_val = (self.z - TOP_LEFT_Y) / (BOTTOM_RIGHT_Y - TOP_LEFT_Y) * 3840

        pyautogui.moveTo(x_val, y_val, _pause=False)

        #pyautogui.moveTo(2000, 10, _pause=False)

        logger.debug("Move Mouse to")
        logger.debug(x_val)
        logger.debug(y_val)

class ReflectoahListener(roypy.IDepthDataListener):
    MOUSE_DOWN = False

    def __init__(self, redis):
        super(ReflectoahListener, self).__init__()
        self.redis = redis
        self.last_click = time.time()

    def onNewData(self, data):
        logger.debug("got new data")

        x_max_point_index, x_max = None, -np.inf
        for i in range(data.getNumPoints()):
            # 3 conditions for clipping:
            # 1. if z>700  => NAN
            # 2. if y>220 or y<-220 => NAN
            # 3. if x<-5 => NAN

            if data.getDepthConfidence(i) > 0 and \
                    not (data.getX(i) > MIN_X_DIST or data.getX(i) < MAX_X_DIST
                         or data.getY(i) > MIN_Y_DIST or data.getY(i) < MAX_Y_DIST or data.getZ(i) > MIN_Z_DIST):
                x = data.getX(i)

                # check if point is closest point to mirror
                if x > x_max:
                    x_max = x
                    x_max_point_index = i

        logger.debug("x_max: {}\nx_max_ind: {}\n\n".format(x_max, x_max_point_index))

        if x_max_point_index:

            x_y_values = []

            width=10
            index = x_max_point_index

            for r in range(-width, width + 1):
                i = index + r * data.width
                for c in range(i - width, i + width + 1):
                    if data.getDepthConfidence(c) > 0 and not (
                            data.getX(c) > MIN_X_DIST or data.getX(c) < MAX_X_DIST or data.getY(
                        c) > MIN_Y_DIST or data.getY(c) < MAX_Y_DIST or data.getZ(c) > MIN_Z_DIST):
                        x_y_values.append([data.getY(c), data.getZ(c)])
            roi = np.array(x_y_values)



            # roi = self.find_roi_around_point_by_index(x_max_point_index, data, width=10)
            medium_roi = roi[:, 0].mean(), roi[:, 1].mean()

            # send message to queue
            self.redis.set('mouse_move', json.dumps(medium_roi))
            #self.channel.basic_publish(exchange='', routing_key='mouse_move', body=json.dumps(medium_roi))
            
            #mmt = MouseMoveThread(medium_roi[0], medium_roi[1])
            #mmt.start()

            # check if user is clicking
            if x_max > CLICK_THRESHOLD and not self.MOUSE_DOWN:
                self.click()
            if x_max < (CLICK_THRESHOLD*1.1):
                self.MOUSE_DOWN = False


    def find_roi_around_point_by_index(self, index, data, width=10):
        x_y_values = []

        for r in range(-width, width + 1):
            i = index + r * data.width
            for c in range(i - width, i + width + 1):
                if data.getDepthConfidence(c) > 0 and not (
                        data.getX(c) > MIN_X_DIST or data.getX(c) < MAX_X_DIST or data.getY(
                    c) > MIN_Y_DIST or data.getY(c) < MAX_Y_DIST or data.getZ(c) > MIN_Z_DIST):
                    x_y_values.append([data.getY(c), data.getZ(c)])

        return np.array(x_y_values)

    def click(self):
        pyautogui.click()
        self.MOUSE_DOWN = True
        logger.info("Mouse Down")

    def paint(self, data):
        """
        Called in the main thread, with data containing one of the items that was added to the queue in onNewData
        :param data:
        :return:
        """
        pass
        # create a figure and show the raw data
        # plt.figure(1)
        # plt.imshow(data)
        #
        # plt.show(block=False)
        # plt.draw()
        #
        # # this pause is needed to ensure the drawing for some backends
        # plt.pause(0.001)

    def move_mouse_by_coords(self, x, y, z):
        logger.debug("move mouse2")

        x_val = (y - TOP_LEFT_X) / (BOTTOM_RIGHT_X - TOP_LEFT_X) * screen_size[0]
        y_val = (z - TOP_LEFT_Y) / (BOTTOM_RIGHT_Y - TOP_LEFT_Y) * screen_size[1]

        #x_val = (y - TOP_LEFT_X) / (BOTTOM_RIGHT_X - TOP_LEFT_X) * 2160
        #y_val = (z - TOP_LEFT_Y) / (BOTTOM_RIGHT_Y - TOP_LEFT_Y) * 3480

        #pyautogui.moveTo(x_val, y_val, _pause=False)
        pyautogui.moveTo(10, 10, _pause=False)

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
