    #ifndef AUDIO_SUPPRESSER_H
#define AUDIO_SUPPRESSER_H

#include <cstdint>
#include <vector>

namespace Model
{

class AudioSuppresser
{
public:
    static void suppressSources(const std::vector<int> &indexToSuppress, uint8_t* audioBuf, const int bufferLength);

private:
    static void createMaskFromIndex(const int index, uint8_t *mask, const int maskLength);
};

}

#endif // AUDIO_SUPPRESSER_H
