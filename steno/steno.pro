# If you want to compile without using CUDA (Everything will run on cpu)
# no_cuda(default) -> compile wihtout cuda, cuda -> compile with cuda
compilation = no_cuda

# Uncomment if you want to cross-compile for Jetson Tx2
target = jetson_tx2

# --------------------------------------------------------------------------------
# !! Do NOT manual modify beyond this point, unless you know what you are doing !!
# --------------------------------------------------------------------------------

contains(target, jetson_tx2) {
    architecture = aarch64
    compilation = cuda
} else {
    architecture = $$QMAKE_HOST.arch
}

message("Compilation  : "$$compilation)
message("Architecture : "$$architecture)

! include (3rd/v4l2.pri) {
    error( "v4l2.pri not found in 3rd" )
}

! include (3rd/darknet.pri) {
    error( "darknet.pri not found in 3rd" )
}

QT += core gui network widgets multimedia multimediawidgets

CONFIG += c++14

DESTDIR = bin
OBJECTS_DIR = bin
MOC_DIR = bin
UI_DIR = bin

# Add 3rd party library dependency
LIBS += $$V4L2_LIBS $$DARKNET_LIBS -lpulse-simple -lpulse -lpthread

INCLUDEPATH *= src

SOURCES += \
    src/main.cpp \
    src/model/audio_suppresser/audio_suppresser.cpp \
    src/model/classifier/classifier.cpp \
    src/model/config/base_config.cpp \
    src/model/config/config.cpp \
    src/model/media/media.cpp \
    src/model/media_player/media_player.cpp \
    src/model/media_player/subtitles/srt_file.cpp \
    src/model/media_player/subtitles/subtitles.cpp \
    src/model/stream/utils/images/image_drawing.cpp \
    src/model/stream/video/output/default_virtual_camera_output.cpp \
    src/model/transcription/transcription.cpp \
    src/model/utils/filesutil.cpp \
    src/model/utils/time.cpp \
    src/model/stream/audio/file/raw_file_audio_sink.cpp \
    src/model/stream/audio/odas/odas_audio_source.cpp \
    src/model/stream/audio/odas/odas_client.cpp \
    src/model/stream/audio/odas/odas_position_source.cpp \
    src/model/stream/audio/pulseaudio/pulseaudio_sink.cpp \
    src/model/stream/audio/source_position.cpp \
    src/model/stream/media_thread.cpp \
    src/model/stream/stream.cpp \
    src/model/stream/utils/alloc/heap_object_factory.cpp \
    src/model/stream/utils/images/image_converter.cpp \
    src/model/stream/utils/images/image_format.cpp \
    src/model/stream/utils/images/stb/stb_image.cpp \
    src/model/stream/utils/images/stb/stb_image_write.cpp \
    src/model/stream/utils/math/angle_calculations.cpp \
    src/model/stream/utils/math/geometry_utils.cpp \
    src/model/stream/utils/time/time_utils.cpp \
    src/model/stream/utils/threads/thread.cpp \
    src/model/stream/video/detection/base_darknet_detector.cpp \
    src/model/stream/video/detection/darknet_detector.cpp \
    src/model/stream/video/detection/detection_thread.cpp \
    src/model/stream/video/detection/detector_mock.cpp \
    src/model/stream/video/dewarping/cpu_darknet_fisheye_dewarper.cpp \
    src/model/stream/video/dewarping/cpu_dewarping_mapping_filler.cpp \
    src/model/stream/video/dewarping/cpu_fisheye_dewarper.cpp \
    src/model/stream/video/dewarping/dewarping_helper.cpp \
    src/model/stream/video/impl/implementation_factory.cpp \
    src/model/stream/video/input/base_camera_reader.cpp \
    src/model/stream/video/input/camera_reader.cpp \
    src/model/stream/video/input/vc_camera_reader.cpp \
    src/model/stream/video/input/image_file_reader.cpp \
    src/model/stream/video/output/image_file_writer.cpp \
    src/model/stream/video/output/virtual_camera_output.cpp \
    src/model/stream/video/video_stabilizer.cpp \
    src/model/stream/video/virtualcamera/display_image_builder.cpp \
    src/model/stream/video/virtualcamera/virtual_camera_manager.cpp \
    src/view/components/side_bar.cpp \
    src/view/components/side_bar_item.cpp \
    src/view/components/top_bar.cpp \
    src/view/mainwindow.cpp \
    src/view/views/conference_view.cpp \
    src/view/views/media_player_view.cpp \
    src/view/views/settings_view.cpp

HEADERS += \
    src/model/app_config.h \
    src/model/audio_suppresser/audio_suppresser.h \
    src/model/classifier/classifier.h \
    src/model/config/base_config.h \
    src/model/config/config.h \
    src/model/media/media.h \
    src/model/media_player/i_media_player.h \
    src/model/media_player/media_player.h \
    src/model/media_player/subtitles/srt_file.h \
    src/model/media_player/subtitles/subtitle_item.h \
    src/model/media_player/subtitles/subtitles.h \
    src/model/recorder/i_recorder.h \
    src/model/stream/audio/audio_config.h \
    src/model/stream/stream_config.h \
    src/model/stream/utils/images/image_drawing.h \
    src/model/transcription/transcription.h \
    src/model/transcription/transcription_config.h \
    src/model/stream/video/output/default_virtual_camera_output.h \
    src/model/utils/filesutil.h \
    src/model/utils/observer/i_observer.h \
    src/model/utils/observer/i_subject.h \
    src/model/utils/time.h \
    src/model/stream/audio/i_audio_sink.h \
    src/model/stream/audio/i_audio_source.h \
    src/model/stream/audio/i_position_source.h \
    src/model/stream/audio/file/raw_file_audio_sink.cpp \
    src/model/stream/audio/odas/odas_audio_source.h \
    src/model/stream/audio/odas/odas_client.h \
    src/model/stream/audio/odas/odas_position_source.h \
    src/model/stream/audio/pulseaudio/pulseaudio_sink.h \
    src/model/stream/audio/source_position.h \
    src/model/stream/i_stream.h \
    src/model/stream/media_thread.h \
    src/model/stream/stream.h \
    src/model/stream/utils/alloc/cuda/device_cuda_object_factory.h \
    src/model/stream/utils/alloc/cuda/managed_memory_cuda_object_factory.h \
    src/model/stream/utils/alloc/cuda/zero_copy_cuda_object_factory.h \
    src/model/stream/utils/alloc/heap_object_factory.h \
    src/model/stream/utils/alloc/i_object_factory.h \
    src/model/stream/utils/array_utils.h \
    src/model/stream/utils/images/cuda/cuda_image_converter.h \
    src/model/stream/utils/images/i_image_converter.h \
    src/model/stream/utils/images/image_converter.h \
    src/model/stream/utils/images/image_format.h \
    src/model/stream/utils/images/images.h \
    src/model/stream/utils/images/stb/stb_image.h \
    src/model/stream/utils/images/stb/stb_image_write.h \
    src/model/stream/utils/macros/packing.h \
    src/model/stream/utils/math/angle_calculations.h \
    src/model/stream/utils/math/geometry_utils.h \
    src/model/stream/utils/math/helpers.h \
    src/model/stream/utils/math/math_constants.h \
    src/model/stream/utils/models/bounding_box.h \
    src/model/stream/utils/models/circular_buffer.h \
    src/model/stream/utils/models/dim2.h \
    src/model/stream/utils/models/dim3.h \
    src/model/stream/utils/models/dual_buffer.h \
    src/model/stream/utils/models/point.h \
    src/model/stream/utils/models/rectangle.h \
    src/model/stream/utils/models/spherical_angle_box.h \
    src/model/stream/utils/models/spherical_angle_rect.h \
    src/model/stream/utils/threads/atomicops.h \
    src/model/stream/utils/threads/lock_triple_buffer.h \
    src/model/stream/utils/threads/readerwriterqueue.h \
    src/model/stream/utils/threads/sync/cuda_synchronizer.h \
    src/model/stream/utils/threads/sync/i_synchronizer.h \
    src/model/stream/utils/threads/sync/nop_synchronizer.h \
    src/model/stream/utils/threads/thread.h \
    src/model/stream/utils/time/time_utils.h \
    src/model/stream/utils/time/timer.h \
    src/model/stream/utils/vector_utils.h \
    src/model/stream/video/detection/base_darknet_detector.h \
    src/model/stream/video/detection/cuda/cuda_darknet_detector.h \
    src/model/stream/video/detection/darknet_detector.h \
    src/model/stream/video/detection/detection_thread.h \
    src/model/stream/video/detection/detector_mock.h \
    src/model/stream/video/detection/darknet_config.h \
    src/model/stream/video/detection/i_detector.h \
    src/model/stream/video/dewarping/cpu_darknet_fisheye_dewarper.h \
    src/model/stream/video/dewarping/cpu_dewarping_mapping_filler.h \
    src/model/stream/video/dewarping/cpu_fisheye_dewarper.h \
    src/model/stream/video/dewarping/cuda/cuda_darknet_fisheye_dewarper.h \
    src/model/stream/video/dewarping/cuda/cuda_dewarping_mapping_filler.h \
    src/model/stream/video/dewarping/cuda/cuda_fisheye_dewarper.h \
    src/model/stream/video/dewarping/dewarping_helper.h \
    src/model/stream/video/dewarping/i_detection_fisheye_dewarper.h \
    src/model/stream/video/dewarping/i_fisheye_dewarper.h \
    src/model/stream/video/dewarping/models/dewarping_config.h \
    src/model/stream/video/dewarping/models/dewarping_mapping.h \
    src/model/stream/video/dewarping/models/dewarping_parameters.h \
    src/model/stream/video/dewarping/models/donut_slice.h \
    src/model/stream/video/dewarping/models/linear_pixel_filter.h \
    src/model/stream/video/impl/implementation_factory.h \
    src/model/stream/video/input/base_camera_reader.h \
    src/model/stream/video/input/camera_reader.h \
    src/model/stream/video/input/vc_camera_reader.h \
    src/model/stream/video/input/cuda/cuda_camera_reader.h \
    src/model/stream/video/input/cuda/vc_cuda_camera_reader.h \
    src/model/stream/video/input/cuda/cuda_image_file_reader.h \
    src/model/stream/video/input/image_file_reader.h \
    src/model/stream/video/input/i_video_input.h \
    src/model/stream/video/output/image_file_writer.h \
    src/model/stream/video/output/i_video_output.h \
    src/model/stream/video/output/virtual_camera_output.h \
    src/model/stream/video/video_config.h \
    src/model/stream/video/video_stabilizer.h \
    src/model/stream/video/virtualcamera/display_image_builder.h \
    src/model/stream/video/virtualcamera/virtual_camera.h \
    src/model/stream/video/virtualcamera/virtual_camera_manager.h \
    src/view/components/colors.h \
    src/view/components/side_bar.h \
    src/view/components/side_bar_item.h \
    src/view/components/top_bar.h \
    src/view/mainwindow.h \
    src/view/views/abstract_view.h \
    src/view/views/conference_view.h \
    src/view/views/media_player_view.h \
    src/view/views/settings_view.h \
    src/model/stream/utils/cuda_utils.cuh \
    src/model/stream/utils/math/cuda_helpers.cuh \
    src/model/stream/video/dewarping/cuda/cuda_dewarping_helper.cuh

CUDA_SOURCES += \
    src/model/stream/utils/alloc/cuda/device_cuda_object_factory.cu \
    src/model/stream/utils/alloc/cuda/managed_memory_cuda_object_factory.cu \
    src/model/stream/utils/alloc/cuda/zero_copy_cuda_object_factory.cu \
    src/model/stream/utils/images/cuda/cuda_image_converter.cu \
    src/model/stream/utils/threads/sync/cuda_synchronizer.cu \
    src/model/stream/video/detection/cuda/cuda_darknet_detector.cu \
    src/model/stream/video/dewarping/cuda/cuda_darknet_fisheye_dewarper.cu \
    src/model/stream/video/dewarping/cuda/cuda_dewarping_helper.cu \
    src/model/stream/video/dewarping/cuda/cuda_dewarping_mapping_filler.cu \
    src/model/stream/video/dewarping/cuda/cuda_fisheye_dewarper.cu \
    src/model/stream/video/input/cuda/cuda_camera_reader.cu \
    src/model/stream/video/input/cuda/vc_cuda_camera_reader.cu \
    src/model/stream/video/input/cuda/cuda_image_file_reader.cu

contains(compilation, no_cuda) {
    DEFINES += NO_CUDA
} else {

    HOST_CUDA_DIR = /usr/local/cuda-10.0
    TARGET_CUDA_DIR = $$[QT_SYSROOT]/usr/local/cuda-10.0

    INCLUDEPATH *= $$TARGET_CUDA_DIR/include
    LIBS += -L$$TARGET_CUDA_DIR/lib64/stubs -lcuda -L$$TARGET_CUDA_DIR/lib64 -lcudart -lcurand -lcublas
    NVCC = $$HOST_CUDA_DIR/bin/nvcc

    cudaIntr.input = CUDA_SOURCES
    cudaIntr.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.o
    cudaIntr.commands = $$NVCC -ccbin $$QMAKE_CXX -dc --compiler-options \"$(CFLAGS)\" $(INCPATH) $$LIBS -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cudaIntr.variable_out = CUDA_OBJ
    cudaIntr.variable_out += OBJECTS
    cudaIntr.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cudaIntr

    CUDA_LINK = bin/cuda_fisheye_dewarper.o

    cuda.input = CUDA_OBJ
    cuda.output =  $$OBJECTS_DIR/cuda_link.o
    cuda.commands = $$NVCC -dlink -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    cuda.CONFIG = combine
    QMAKE_EXTRA_COMPILERS += cuda
}

FORMS += \
    src/view/gui/conference_view.ui \
    src/view/gui/mainwindow.ui \
    src/view/gui/media_player_view.ui \
    src/view/gui/settings_view.ui \
    src/view/gui/side_bar.ui \
    src/view/gui/side_bar_item.ui \
    src/view/gui/top_bar.ui

RESOURCES += \
    resources/resources.qrc

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /home/nvidia/rendezvous/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
