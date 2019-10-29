DARKNET_DIR = 3rd/darknet
INCLUDEPATH *= $$DARKNET_DIR/include

architecture = $$QMAKE_HOST.arch

contains(architecture, aarch64) {
    DARKNET_LIBS = -L$$DARKNET_DIR/lib/aarch64 -ldarknet
}

contains(architecture, x86_64) {
    DARKNET_LIBS = -L$$DARKNET_DIR/lib/x86_64 -ldarknet
}
