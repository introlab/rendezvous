DARKNET_DIR = 3rd/darknet
INCLUDEPATH *= $$DARKNET_DIR/include
DARKNET_LIBS = -L$$DARKNET_DIR/lib/$$QMAKE_HOST.arch -ldarknet
