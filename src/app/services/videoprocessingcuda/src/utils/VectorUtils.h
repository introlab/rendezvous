#ifndef VECTOR_UTILS_H
#define VECTOR_UTILS_H

#include <algorithm>
#include <functional>
#include <vector>

template<typename T, typename F>
void removeElementsAndPack(std::vector<T>& vec, const F& checkFunc)
{
    vec.erase(std::remove_if(vec.begin(), vec.end(), checkFunc), vec.end());
}

#endif //!VECTOR_UTILS_H