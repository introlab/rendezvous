QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets multimedia multimediawidgets

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
    model/media_player.cpp \
    model/settings.cpp \
    view/mainwindow.cpp \
    view/components/sidebar.cpp \
    view/views/conference_view.cpp \
    view/views/media_player_view.cpp \
    view/views/recording_view.cpp \
    view/views/settings_view.cpp \
    view/views/transcription_view.cpp

HEADERS += \
    model/i_media_player.h \
    model/i_settings.h \
    model/media_player.h \
    model/settings.h \
    view/mainwindow.h \
    view/components/sidebar.h \
    view/views/abstract_view.h \
    view/views/conference_view.h \
    view/views/media_player_view.h \
    view/views/recording_view.h \
    view/views/settings_view.h \
    view/views/struct_appclication_settings.h \
    view/views/transcription_view.h

FORMS += \
    view/gui/conference_view.ui \
    view/gui/mainwindow.ui \
    view/gui/media_player_view.ui \
    view/gui/recording_view.ui \
    view/gui/settings_view.ui \
    view/gui/transcription_view.ui

DISTFILES += \
    view/stylesheets/globalStylesheet.qss

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
