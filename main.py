import argparse
import json
import logging
import queue
import time

import redis
import pyautogui

from Listener.MyListener import MyListener
from Listener.ReflectoahListener import ReflectoahListener
from utils.roypy_platform_utils import PlatformHelper
from utils.roypy_sample_utils import CameraOpener, add_camera_opener_options
from utils.sample_camera_info import print_camera_info

logger = logging.getLogger("reflectoah")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())




pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0


frames = []

r = redis.Redis(host='localhost')

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
    parser = argparse.ArgumentParser(usage=__doc__)
    add_camera_opener_options(parser)
    parser.add_argument("--seconds", type=int, default=15, help="duration to capture data")
    options = parser.parse_args()
    opener = CameraOpener(options)
    cam = opener.open_camera()

    z_queue = queue.Queue()
    gray_queue = queue.Queue()
    l = MyListener(z_queue)

    cam.registerDataListener(l)
    cam.startCapture()

    process_event_queue(z_queue, gray_queue, painter=l, seconds=10)
    # process_event_queue(z_queue, gray_queue)
    cam.stopCapture()


def sample_mouse_move():
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser(usage=__doc__)
    add_camera_opener_options(parser)
    parser.add_argument("--seconds", type=int, default=15, help="duration to capture data")
    options = parser.parse_args()
    opener = CameraOpener(options)
    cam = opener.open_camera()

    q = queue.Queue()
    l = ReflectoahListener(r)
    # status = cam.getLensParameters(lensParameters)
    # l.setLensParameters(lensParameters)

    logger.info("Getting use cases for cam:")
    for uc_i in range(len(cam.getUseCases())):
        logger.info(cam.getUseCases()[uc_i])

    logger.info("\nSetting use case for cam to {}".format(cam.getUseCases()[-2]))
    cam.setUseCase(cam.getUseCases()[-2])
    # cam.setUseCase(cam.getUseCases()[0])

    cam.registerDataListener(l)
    cam.startCapture()

    # process_event_queue(q, None, painter=l)
    time.sleep(20)
    cam.stopCapture()
    connection.close()


def process_event_queue(z_queue, gray_queue, painter=None, seconds=150):
    # create a loop that will run for the given amount of time
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            # try to retrieve an item from the queue
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit

            zImage = z_queue.get(True) if z_queue else None
        except queue.Empty:
            # this will be thrown when the timeout is hit
            pass
            # break
        else:
            if painter:
                painter.paint(zImage)
            elif zImage is not None:
                import cv2
                cv2.imshow("Depth", zImage)
                cv2.waitKey(1)
            else:
                time.sleep(0.001)

        # try:
        #     # try to retrieve an item from the queue
        #     # this will block until an item can be retrieved
        #     # or the timeout of 1 second is hit
        #     grayImage = gray_queue.get(True, 1)
        # except queue.Empty:
        #     # this will be thrown when the timeout is hit
        #     pass
        #     # break
        # else:
        #     if painter:
        #         pass
        #     else:
        #         cv2.imshow("Gray", grayImage)
        #         cv2.waitKey(1)


if __name__ == '__main__':
    # sample_retrieve_data()
    # sample_open_cv()
    sample_mouse_move()

    # logger.info("Mean running time: {}".format(np.array(times).mean()))
