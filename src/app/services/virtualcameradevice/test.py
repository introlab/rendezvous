import time
from interface.virtual_camera_device import VirtualCameraDevice

def main():
    virtualCameraDevice = VirtualCameraDevice(videoDevice="/dev/video1", format=0, width=640, height=480, fps=15)

    virtualCameraDevice.test()

    time.sleep(5)

    virtualCameraDevice.__del__


if __name__ == '__main__':
	main()