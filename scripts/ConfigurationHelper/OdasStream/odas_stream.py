import subprocess
import json
import math as mt
from threading import Thread
from time import sleep

import numpy as np


class OdasStream:

    def __init__(self, odasPath, configPath):
        self.odasPath = odasPath
        self.configPath = configPath


    # Create and start thread for capturing odas stream
    def start(self):
        print('Beginning of the odas coordinates transmission...')
        Thread(target=self.__run, args=()).start()


    # Stop odas Thread
    def stop(self):
        print('Stopping odas stream...')


    def __run(self):

        stdout = []
        stdoutobj = []

        if (self.odasPath and self.configPath):
            print('ODAS stream starting...')

            result = subprocess.Popen([self.odasPath, '-c', self.configPath], shell=True, stdout=subprocess.PIPE)
            subprocess.call([self.odasPath, '-c', self.configPath])

            # Need to check if device detected
            while True:
                line = result.stdout.readline().decode('UTF-8')
                stdoutobj.append(line)

                if len(stdoutobj) > 8:
                    stdout.extend(stdoutobj)
                    stdoutobj.clear()

                if stdout:
                    textoutput = '\n'.join(stdout)
                    self.__parseJsonStream(textoutput)
                    stdout.clear()
                
                sleep(0.01) # sleep for 1ms


    def __parseJsonStream(self, jsonText):

        self.stream = []
        self.source1 = []
        self.source2 = []
        self.source3 = []
        self.source4 = []
        self.source = np.array([])
        sources = np.ndarray

        parsed_json = json.loads(jsonText)
        timeStamp = parsed_json['timeStamp']
        src = parsed_json['src']

        self.stream.append([timeStamp, src])

        id1 = parsed_json['src'][0]['id']
        tag1 = parsed_json['src'][0]['tag']
        x1 = parsed_json['src'][0]['x']
        y1 = parsed_json['src'][0]['y']
        z1 = parsed_json['src'][0]['z']
        activity1 = parsed_json['src'][0]['activity']
        isActive1 = not (x1 == 0 and y1 == 0 and z1 == 0)

        if isActive1:
            xf1, yf1 = self.__calculate2dcoord(x1, y1)
            self.source1 = np.array([xf1, yf1])
            self.source = np.append(self.source, xf1)
            self.source = np.append(self.source, yf1)


        id2 = parsed_json['src'][1]['id']
        tag2 = parsed_json['src'][1]['tag']
        x2 = parsed_json['src'][1]['x']
        y2 = parsed_json['src'][1]['y']
        z2 = parsed_json['src'][1]['z']
        activity2 = parsed_json['src'][1]['activity']
        isActive2 = not (x2 == 0 and y2 == 0 and z2 == 0)

        if isActive2:
            xf2, yf2 = self.__calculate2dcoord(x2, y2)
            self.source2 = np.array([xf2, yf2])
            self.source = np.append(self.source, xf2)
            self.source = np.append(self.source, yf2)


        id3 = parsed_json['src'][2]['id']
        tag3 = parsed_json['src'][2]['tag']
        x3 = parsed_json['src'][2]['x']
        y3 = parsed_json['src'][2]['y']
        z3 = parsed_json['src'][2]['z']
        activity3 = parsed_json['src'][2]['activity']
        isActive3 = not (x3 == 0 and y3 == 0 and z3 == 0)

        if isActive3:
            xf3, yf3 = self.__calculate2dcoord(x3, y3)
            self.source3 = np.array([xf3, yf3])
            self.source = np.append(self.source, xf3)
            self.source = np.append(self.source, yf3)


        id4 = parsed_json['src'][3]['id']
        tag4 = parsed_json['src'][3]['tag']
        x4 = parsed_json['src'][3]['x']
        y4 = parsed_json['src'][3]['y']
        z4 = parsed_json['src'][3]['z']
        activity4 = parsed_json['src'][3]['activity']
        isActive4 = not (x4 == 0 and y4 == 0 and z4 == 0)

        if isActive4:
            xf4, yf4 = self.__calculate2dcoord(x4, y4)
            self.source4 = np.array([xf4, yf4])
            self.source = np.append(self.source, xf4)
            self.source = np.append(self.source, yf4)

        if self.source.size != 0:
            print(self.source)

    # Unwarp Odas coordinates to panoramic coordinates
    def __calculate2dcoord(self, xi, yi):

        # The largest radius of your circle if your input image is rectangular
        width = 640
        half_width = 320

        x = (xi + 1) * half_width  # + 640
        y = (yi + 1) * half_width  # + 640

        if x == half_width:
            x = half_width + 0.1

        if y == half_width:
            y = half_width + 0.1

        xcenter = half_width
        ycenter = half_width

        R1 = 350 - xcenter  # To change according to the image unwarping parameters
        R2 = width - xcenter

        wd = 2.0 * ((R2 + R1) / 2) * mt.pi
        hd = (R2 - R1)

        theta = (mt.atan((x - xcenter) / (y - ycenter)))

        # Other half
        if y < half_width:
            theta = theta + mt.pi

        r = ((x - xcenter) / mt.sin(theta))

        xfin = (float(wd) / (2 * mt.pi)) * theta
        yfin = ((r - R1) / (R2 - R1)) * float(hd)
        yfin = yfin - 80     # Correct the height difference between the 0 coord. of Odas and the 0 coord. of the camera

        if xfin < 0:
            xfin = wd + xfin

        return xfin, yfin

