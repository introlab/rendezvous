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

%apply (int DIM1, int DIM2, int DIM3, unsigned char * IN_ARRAY3) { (int width, int height, int channels, unsigned char * fisheyeImage) };
%apply (int DIM1, int DIM2, int DIM3, unsigned char * IN_ARRAY3) { (int width, int height, int channels, unsigned char * dewarpedImage) };
%include "src/FisheyeDewarping.h"
%include "../include/models/DonutSlice.h"
%include "../include/models/DewarpingParameters.h"