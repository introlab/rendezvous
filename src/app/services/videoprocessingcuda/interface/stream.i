%module stream

%{
    #define SWIG_FILE_WITH_INIT
    #include "src/Stream.h"
    #include "../src/dewarping/models/DewarpingConfig.h"
    #include "../src/streaming/input/CameraConfig.h"
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
%include "src/Stream.h"
%include "../src/dewarping/models/DewarpingConfig.h"
%include "../src/streaming/input/CameraConfig.h"