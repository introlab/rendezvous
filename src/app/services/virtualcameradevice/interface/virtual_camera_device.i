%module virtual_camera_device

%{
    #define SWIG_FILE_WITH_INIT
    #include "src/VirtualCameraDevice.h"
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

%apply (unsigned char * IN_ARRAY3, int DIM1, int DIM2, int DIM3) { (unsigned char * buffer, int height, int width, int channels) };

%include "src/VirtualCameraDevice.h"
