V4L2_DIR = 3rd/v4l2
INCLUDEPATH *= $$V4L2_DIR/include

architecture = $$QMAKE_HOST.arch

contains(architecture, aarch64) {
    V4L2_LIBS = -L$$V4L2_DIR/lib/aarch64 -lv4l2
}

contains(architecture, x86_64) {
    V4L2_LIBS = -L$$V4L2_DIR/lib/x86_64-lv4l2
}
