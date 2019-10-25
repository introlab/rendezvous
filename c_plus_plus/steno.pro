QT += core gui network widgets multimedia multimediawidgets

CONFIG += c++14

DESTDIR = bin
OBJECTS_DIR = bin
MOC_DIR = bin
UI_DIR = bin

INCLUDEPATH += src/

LIBS += -lpulse-simple -lpulse

SOURCES += \
    src/main.cpp \
    src/model/media_player/media_player.cpp \
    src/model/network/local_socket_server.cpp \
    src/model/audio/odas/odas_audio_source.cpp \
    src/model/audio/odas/odas_position_source.cpp \
    src/model/audio/source_position.cpp \
    src/model/audio/pulseaudio/pulseaudio_sink.cpp \
    src/model/recorder/recorder.cpp \
    src/model/utils/spherical_angle_converter.cpp \
    src/model/settings/settings.cpp \
    src/view/mainwindow.cpp \
    src/view/components/sidebar.cpp \
    src/view/views/local_conference_view.cpp \
    src/view/views/media_player_view.cpp \
    src/view/views/online_conference_view.cpp \
    src/view/views/settings_view.cpp

HEADERS += \
    src/model/media_player/i_media_player.h \
    src/model/media_player/media_player.h \
    src/model/network/i_socket_server.h \
    src/model/network/local_socket_server.h \
    src/model/audio/i_audio_sink.h \
    src/model/audio/i_audio_source.h \
    src/model/audio/i_position_source.h \
    src/model/audio/source_position.h \
    src/model/audio/pulseaudio/pulseaudio_sink.h \
    src/model/audio/odas/odas_audio_source.h \
    src/model/audio/odas/odas_position_source.h \
    src/model/recorder/i_recorder.h \
    src/model/recorder/recorder.h \
    src/model/utils/spherical_angle_converter.h \
    src/model/settings/i_settings.h \
    src/model/settings/settings.h \
    src/model/settings/settings_constants.h \
    src/view/mainwindow.h \
    src/view/components/sidebar.h \
    src/view/views/abstract_view.h \
    src/view/views/local_conference_view.h \
    src/view/views/media_player_view.h \
    src/view/views/online_conference_view.h \
    src/view/views/settings_view.h

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
