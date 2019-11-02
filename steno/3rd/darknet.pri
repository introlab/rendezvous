DARKNET_DIR = 3rd/darknet
INCLUDEPATH *= $$DARKNET_DIR/include

architecture = $$QMAKE_HOST.arch

contains(architecture, aarch64) {
    DARKNET_LIBS = -L$$DARKNET_DIR/lib/aarch64 -ldarknet
}

contains(architecture, x86_64) {
    DARKNET_LIBS = -L$$DARKNET_DIR/lib/x86_64 -ldarknet

    nvcc_path = $$system(which nvcc 2)
    isEmpty(nvcc_path) {
        DARKNET_LIBS = -L$$DARKNET_DIR/lib/x86_64 -ldarknet
    }
    else {
        DARKNET_LIBS = -L$$DARKNET_DIR/lib/x86_64_cuda -ldarknet
    }
}
