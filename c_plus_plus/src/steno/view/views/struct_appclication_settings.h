#ifndef STRUCT_APPCLICATION_SETTINGS_H
#define STRUCT_APPCLICATION_SETTINGS_H

#include <string>


namespace View
{
    struct ApplicationSettingsStruct
    {
        struct General
        {
            std::string defaultConfigurationFilePath= "defaultConfigurationFilePath";
        }general;

        struct Conference
        {
            std::string cameraConfigurationFilePath = "cameraConfigurationFilePath";    // .json
            std::string microConfigurationFilePath = "microConfigurationFilePath";     // .cfg
            std::string odasLibraryFilePath = "odasLibraryFilePath";            // folder path

            enum FaceDetectionMethods
            {
                OPENCV_DNN,
                OPENcv_HAAR_CASCADE,
                YOLOV3
            };
        }conference;

        struct Transcription
        {
            std::string googleServiceAcountFilePath = "googleServiceAcountFilePath";  //.json

            int MIN_CHANEL_COUNT = 1;
            int MAX_CHANEL_COUNT = 254;

            enum Encoding
            {
                ENCODING_UNSPECIFIED,
                FLAC,
                AMR,
                AMR_WB,
                LINEAR16,
                OGG_OPUS,
                SPEEX_WITH_HEADER_BYTE
            };

            enum Languages
            {
                FR_CA,
                EN_CA
            };

            enum Model
            {
                DEFAULT_MODEL,
                COMMAND_AND_SEARCH,
                PHONE_CALL,
                VIDEO
            }model;

            enum SampleRates
            {
                FREQ_8000_HZ,
                FREQ_11025_HZ,
                FREQ_22050_HZ,
                FREQ_32000_HZ,
                FREQ_44100_HZ,
                FREQ_48000_HZ
            }samplerates;
        }transcription;
    };
}

#endif // STRUCT_APPCLICATION_SETTINGS_H
