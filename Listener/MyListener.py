import numpy as np
import matplotlib.pyplot as plt
import roypy


class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.queue = q

    def onNewData(self, data):
        zvalues = []
        for i in range(data.getNumPoints()):
            zvalues.append(data.getZ(i))
        zarray = np.asarray(zvalues)
        p = zarray.reshape(-1, data.width)
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