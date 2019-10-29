QT += core gui network widgets multimedia multimediawidgets

CONFIG += c++14

DESTDIR = bin
OBJECTS_DIR = bin
MOC_DIR = bin
UI_DIR = bin

# If you want to compile without using CUDA (Everything will run on cpu)
compilation = no_cuda

INCLUDEPATH += src $(DARKNET_HOME)/include $(LIBV4L2CPP_HOME)/inc

LIBS += -lpulse-simple -lpulse -lpthread -L$(DARKNET_HOME) -ldarknet -L$(LIBV4L2CPP_HOME) -lv4l2wrapper

SOURCES += \
    src/main.cpp \
    src/model/audio_suppresser/audio_suppresser.cpp \
    src/model/classifier/classifier.cpp \
    src/model/media_player/media_player.cpp \
    src/model/media_player/subtitles/srt_file.cpp \
    src/model/media_player/subtitles/subtitles.cpp \
    src/model/network/local_socket_server.cpp \
    src/model/recorder/recorder.cpp \
    src/model/settings/settings.cpp \
    src/model/utils/time.cpp \
    src/model/stream/audio/file/raw_file_audio_sink.cpp \
    src/model/stream/audio/odas/odas_audio_source.cpp \
    src/model/stream/audio/odas/odas_position_source.cpp \
    src/model/stream/audio/pulseaudio/pulseaudio_sink.cpp \
    src/model/stream/audio/source_position.cpp \
    src/model/stream/stream.cpp \
    src/model/stream/utils/alloc/heap_object_factory.cpp \
    src/model/stream/utils/images/image_converter.cpp \
    src/model/stream/utils/images/image_format.cpp \
    src/model/stream/utils/images/stb/stb_image.cpp \
    src/model/stream/utils/images/stb/stb_image_write.cpp \
    src/model/stream/utils/math/angle_calculations.cpp \
    src/model/stream/utils/math/geometry_utils.cpp \
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
    src/model/stream/video/input/camera_reader.cpp \
    src/model/stream/video/input/image_file_reader.cpp \
    src/model/stream/video/output/image_file_writer.cpp \
    src/model/stream/video/output/virtual_camera_output.cpp \
    src/model/stream/video/video_stabilizer.cpp \
    src/model/stream/video/video_thread.cpp \
    src/model/stream/video/virtualcamera/display_image_builder.cpp \
    src/model/stream/video/virtualcamera/virtual_camera_manager.cpp \
    src/view/components/sidebar.cpp \
    src/view/mainwindow.cpp \
    src/view/views/local_conference_view.cpp \
    src/view/views/media_player_view.cpp \
    src/view/views/online_conference_view.cpp \
    src/view/views/settings_view.cpp

HEADERS += \
    src/model/audio_suppresser/audio_suppresser.h \
    src/model/classifier/classifier.h \
    src/model/media_player/i_media_player.h \
    src/model/media_player/media_player.h \
    src/model/media_player/subtitles/srt_file.h \
    src/model/media_player/subtitles/subtitle_item.h \
    src/model/media_player/subtitles/subtitles.h \
    src/model/network/i_socket_server.h \
    src/model/network/local_socket_server.h \
    src/model/recorder/i_recorder.h \
    src/model/recorder/recorder.h \
    src/model/settings/i_settings.h \
    src/model/settings/settings_constants.h \
    src/model/utils/time.h \
    src/model/settings/settings.h \
    src/model/stream/audio/i_audio_sink.h \
    src/model/stream/audio/i_audio_source.h \
    src/model/stream/audio/i_position_source.h \
    src/model/stream/audio/file/raw_file_audio_sink.cpp \
    src/model/stream/audio/odas/odas_audio_source.h \
    src/model/stream/audio/odas/odas_position_source.h \
    src/model/stream/audio/pulseaudio/pulseaudio_sink.h \
    src/model/stream/audio/source_position.h \
    src/model/stream/i_stream.h \
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
    src/model/stream/utils/time/timer.h \
    src/model/stream/utils/vector_utils.h \
    src/model/stream/video/detection/base_darknet_detector.h \
    src/model/stream/video/detection/cuda/cuda_darknet_detector.h \
    src/model/stream/video/detection/darknet_detector.h \
    src/model/stream/video/detection/detection_thread.h \
    src/model/stream/video/detection/detector_mock.h \
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
    src/model/stream/video/input/camera_reader.h \
    src/model/stream/video/input/cuda/cuda_camera_reader.h \
    src/model/stream/video/input/cuda/cuda_image_file_reader.h \
    src/model/stream/video/input/image_file_reader.h \
    src/model/stream/video/input/i_video_input.h \
    src/model/stream/video/output/image_file_writer.h \
    src/model/stream/video/output/i_video_output.h \
    src/model/stream/video/output/virtual_camera_output.h \
    src/model/stream/video/video_config.h \
    src/model/stream/video/video_stabilizer.h \
    src/model/stream/video/video_thread.h \
    src/model/stream/video/virtualcamera/display_image_builder.h \
    src/model/stream/video/virtualcamera/virtual_camera.h \
    src/model/stream/video/virtualcamera/virtual_camera_manager.h \
    src/view/components/sidebar.h \
    src/view/mainwindow.h \
    src/view/views/abstract_view.h \
    src/view/views/local_conference_view.h \
    src/view/views/media_player_view.h \
    src/view/views/online_conference_view.h \
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
    src/model/stream/video/input/cuda/cuda_image_file_reader.cu

contains(compilation, no_cuda) {
    DEFINES += NO_CUDA
} else {
    INCLUDEPATH += $(CUDA_HOME)/include
    LIBS += -L$(CUDA_HOME)/lib -lcuda -lcudart

    cuda.input = CUDA_SOURCES
    cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.o
    cuda.commands = nvcc -c --compiler-options \"$(CFLAGS)\" $(INCPATH) $$LIBS -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependcy_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

FORMS += \
    src/view/gui/local_conference_view.ui \
    src/view/gui/mainwindow.ui \
    src/view/gui/media_player_view.ui \
    src/view/gui/online_conference_view.ui \
    src/view/gui/settings_view.ui

RESOURCES += \
    resources/resources.qrc

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
