%module fisheye_dewarping

%{
    #define SWIG_FILE_WITH_INIT
    #include "src/FisheyeDewarping.h"
    #include "../include/models/DonutSlice.h"
    #include "../include/models/DewarpingParameters.h"
%}

%include "numpy.i"
%init 
%{
import_array();
%}

%include "typemaps.i"

%apply (unsigned char * IN_ARRAY3, int DIM1, int DIM2, int DIM3) { (unsigned char * fisheyeImage, int height, int width, int channels) };
%apply (unsigned char * IN_ARRAY3, int DIM1, int DIM2, int DIM3) { (unsigned char * dewarpedImageBuffer, int height, int width, int channels) };
%include "src/FisheyeDewarping.h"
%include "../include/models/DonutSlice.h"
%include "../include/models/DewarpingParameters.h"