#ifndef TRANSCRIPTION_CONSTANTS_H
#define TRANSCRIPTION_CONSTANTS_H

namespace Model
{
namespace Transcription
{

enum Encoding
{
    ENCODING_UNSPECIFIED,
    LINEAR16,
    FLAC,
    MULAW,
    AMR,
    AMR_WB,
    OGG_OPUS,
    SPEEX_WITH_HEADER_BYTE
};

inline const char* languageName(Transcription::Encoding encoding)
{
    switch (encoding)
    {
        case Transcription::Encoding::ENCODING_UNSPECIFIED:
            return "Encoding Unspecified";
        case Transcription::Encoding::LINEAR16:
            return "LINEAR16";
        case Transcription::Encoding::FLAC:
            return "Free Lossless Audio Codec";
        case Transcription::Encoding::MULAW:
            return "MULAW";
        case Transcription::Encoding::AMR:
            return "Adaptive Multi-Rate Narrowband";
        case Transcription::Encoding::AMR_WB:
            return "Adaptive Multi-Rate Wideband";
        case Transcription::Encoding::OGG_OPUS:
            return "OggOpus";
        case Transcription::Encoding::SPEEX_WITH_HEADER_BYTE:
            return "Speex Wtih Header Byte";
    }
}

enum Language
{
    FR_CA,
    EN_CA,
    COUNT
};

inline const char* languageName(Transcription::Language language)
{
    switch (language)
    {
        case Transcription::Language::FR_CA:
            return "FR CA";
        case Transcription::Language::EN_CA:
            return "EN CA";
        default:
            return nullptr;
    }
}

enum Model
{
    DEFAULT,
    COMMAND_AND_SEARCH,
    PHONE_CALL,
    VIDEO
};

inline const char* modelName(Transcription::Model model)
{
    switch (model)
    {
        case Transcription::Model::DEFAULT:
            return "Default";
        case Transcription::Model::COMMAND_AND_SEARCH:
            return "Command and search";
        case Transcription::Model::PHONE_CALL:
            return "Phone call";
        case Transcription::Model::VIDEO:
            return "Video";
    }
}

}
}

#endif // TRANSCRIPTION_CONSTANTS_H
