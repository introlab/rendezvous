from enum import Enum, unique

@unique
class FaceDetectionMethods(Enum):
    OPENCV_DNN = 'OpenCV dnn'
    OPENCV_HAAR_CASCADES = 'OpenCV Haar cascades'