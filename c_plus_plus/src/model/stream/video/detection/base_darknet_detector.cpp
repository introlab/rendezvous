#include "base_darknet_detector.h"

namespace Model
{

BaseDarknetDetector::BaseDarknetDetector(const std::string& configFile, const std::string& weightsFile, const std::string& metadataFile)
{
    network_ = load_network(const_cast<char*>(configFile.c_str()), const_cast<char*>(weightsFile.c_str()), 0);
    metadata_ = get_metadata(const_cast<char*>(metadataFile.c_str()));
}

BaseDarknetDetector::~BaseDarknetDetector()
{
    free_network(network_);
}

std::vector<Rectangle> BaseDarknetDetector::detectInImage(const ImageFloat& img)
{
    image darknetImage = convertToDarknetImage(img);
    return detect(darknetImage, 0.5, 0.5, 0.45);
}

Dim2<int> BaseDarknetDetector::getInputImageDim()
{
    return Dim2<int>(network_->w, network_->h);
}

std::vector<Rectangle> BaseDarknetDetector::detect(const image& img, float threshold, float hierThreshold, float nms)
{
    predictImage(network_, img);

    int numDetections = 0;
    detection* detections = get_network_boxes(network_, img.w, img.h, threshold, hierThreshold, nullptr, 0, &numDetections);

    do_nms_obj(detections, numDetections, metadata_.classes, nms);

    std::vector<Rectangle> rects;
    for (int j = 0; j < numDetections; j++)
    {
        for (int i = 0; i < metadata_.classes; i++)
        {
            if (detections[j].prob[i] > 0)
            {
                const box& bbox = detections[j].bbox;
                rects.emplace_back(bbox.x, bbox.y, bbox.w, bbox.y);
            }
        }
    }

    free_detections(detections, numDetections);
    return rects;
}
} // Model
