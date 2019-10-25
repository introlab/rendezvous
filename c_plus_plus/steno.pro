QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets multimedia multimediawidgets

CONFIG += c++14

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

DESTDIR=bin
OBJECTS_DIR=bin
MOC_DIR=bin
UI_DIR=bin

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

DISTFILES += \
    src/view/stylesheets/globalStylesheet.qss

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

