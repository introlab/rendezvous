QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    view/mainwindow.cpp \
    view/components/sidebar.cpp \
    view/views/audio_processing.cpp \
    view/views/conference.cpp \
    view/views/playback.cpp \
    view/views/recording.cpp \
    view/views/settings.cpp \
    view/views/transcription.cpp

HEADERS += \
    view/mainwindow.h \
    view/components/sidebar.h \
    view/views/audio_processing.h \
    view/views/conference.h \
    view/views/playback.h \
    view/views/recording.h \
    view/views/settings.h \
    view/views/transcription.h

FORMS += \
    view/gui/audio_processing.ui \
    view/gui/conference.ui \
    view/gui/mainwindow.ui \
    view/gui/playback.ui \
    view/gui/recording.ui \
    view/gui/settings.ui \
    view/gui/transcription.ui

DISTFILES += \
    view/stylesheets/globalStylesheet.qss

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
