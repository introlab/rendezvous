QT       += core gui

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

#LIBS += -lpulse-simple -lpulse

SOURCES += \
    src/main.cpp \
    #src/model/audio/pulseaudio/pulseaudio_sink.cpp \
    src/model/media_player.cpp \
    src/model/settings/settings.cpp \
    src/view/mainwindow.cpp \
    src/view/components/sidebar.cpp \
    src/view/views/conference_view.cpp \
    src/view/views/media_player_view.cpp \
    src/view/views/recording_view.cpp \
    src/view/views/settings_view.cpp \
    src/view/views/transcription_view.cpp

HEADERS += \
    src/model/audio/i_audio_sink.h \
    #src/model/audio/pulseaudio/pulseaudio_sink.h \
    src/model/i_media_player.h \
    src/model/media_player.h \
    src/model/settings/i_settings.h \
    src/model/settings/settings.h \
    src/model/settings/settings_constants.h \
    src/view/mainwindow.h \
    src/view/components/sidebar.h \
    src/view/views/abstract_view.h \
    src/view/views/conference_view.h \
    src/view/views/media_player_view.h \
    src/view/views/recording_view.h \
    src/view/views/settings_view.h \
    src/view/views/transcription_view.h

FORMS += \
    src/view/gui/conference_view.ui \
    src/view/gui/mainwindow.ui \
    src/view/gui/media_player_view.ui \
    src/view/gui/recording_view.ui \
    src/view/gui/settings_view.ui \
    src/view/gui/transcription_view.ui

DISTFILES += \
    src/view/stylesheets/globalStylesheet.qss

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
