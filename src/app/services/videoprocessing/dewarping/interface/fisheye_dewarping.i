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

%exception { 
    try {
        $action
    } catch (const std::exception& e) {
        SWIG_exception_fail(SWIG_RuntimeError, e.what());
    } catch (...) {
        SWIG_exception_fail(SWIG_RuntimeError, "unknown exception");
    }
}

%include "typemaps.i"

%apply (unsigned char * IN_ARRAY3, int DIM1, int DIM2, int DIM3) { (unsigned char * fisheyeImage, int height, int width, int channels) };
%apply (unsigned char * IN_ARRAY3, int DIM1, int DIM2, int DIM3) { (unsigned char * dewarpedImageBuffer, int height, int width, int channels) };
%include "src/FisheyeDewarping.h"
%include "../include/models/DonutSlice.h"
%include "../include/models/DewarpingParameters.h"