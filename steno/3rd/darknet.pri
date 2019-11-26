DARKNET_DIR = 3rd/darknet
INCLUDEPATH *= $$DARKNET_DIR/include

contains(architecture, aarch64) {
    DARKNET_LIBS = -L$$DARKNET_DIR/lib/aarch64 -ldarknet
    darknet.path = /home/nvidia/rendezvous/$${TARGET}/bin
    darknet.files = $$DARKNET_DIR/lib/aarch64/*
    !isEmpty(darknet.path): INSTALLS += darknet
}

contains(architecture, x86_64) {
    contains(compilation, no_cuda) {
        DARKNET_LIBS = -L$$DARKNET_DIR/lib/x86_64 -ldarknet
    }
    else {
        DARKNET_LIBS = -L$$DARKNET_DIR/lib/x86_64_cuda -ldarknet
    }
}
