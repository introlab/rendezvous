EXECUTABLE=app
SOURCE_FOLDER=src
OUTPUT_FOLDER=bin

INC_PATH=-I$(CUDA_HOME)/include -I$(SOURCE_FOLDER) -I$(DARKNET_HOME)/include -I$(LIBV4L2CPP_HOME)/inc
LIB_PATH=-L$(DARKNET_HOME) -L$(LIBV4L2CPP_HOME)
CLIBS=-lpthread -ldarknet -lv4l2wrapper
CC=g++
NVCC=nvcc
STD_VER=-std=c++14
CFLAGS=-Wall -fPIC $(STD_VER) $(INC_PATH) $(LIB_PATH)

SOURCE_SUBFOLDERS=$(shell find $(SOURCE_FOLDER)/* -type d)
OUTPUT_SUBFOLDERS=$(patsubst $(SOURCE_FOLDER)/%, $(OUTPUT_FOLDER)/%, $(SOURCE_SUBFOLDERS))
SOURCES=$(shell find $(SOURCE_FOLDER) -name "*.cpp")
NVCC_SOURCES=$(shell find $(SOURCE_FOLDER) -name "*.cu")
OBJECTS=$(patsubst $(SOURCE_FOLDER)/%.cpp, $(OUTPUT_FOLDER)/%.o, $(SOURCES))
NVCC_OBJECTS=$(patsubst $(SOURCE_FOLDER)/%.cu, $(OUTPUT_FOLDER)/%.o, $(NVCC_SOURCES))

.PHONY: lib
lib: mkdir_output $(OBJECTS) create_lib interface

.PHONY: lib_no_cuda
lib_no_cuda: CFLAGS+=-DNO_CUDA
lib_no_cuda: mkdir_output $(OBJECTS) create_lib interface_no_cuda

.PHONY: app
app: mkdir_output build_cuda

.PHONY: app_no_cuda
app_no_cuda: CFLAGS+=-DNO_CUDA
app_no_cuda: mkdir_output build

.PHONY: interface
interface: 
	$(MAKE) -C ./interface NVCC_SOURCES="$(NVCC_SOURCES)"

.PHONY: interface_no_cuda
interface_no_cuda: 
	$(MAKE) -C ./interface

.PHONY: create_lib
create_lib: 
	@mkdir -p lib
	ar rvs lib/libStream.a $(OBJECTS)

build: $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(OUTPUT_FOLDER)/$(EXECUTABLE) $(CLIBS)

build_cuda: $(OBJECTS) $(NVCC_OBJECTS)
	$(NVCC) --compiler-options "$(CFLAGS)" $(OBJECTS) $(NVCC_OBJECTS) -o $(OUTPUT_FOLDER)/$(EXECUTABLE) $(CLIBS)

$(OBJECTS): $(OUTPUT_FOLDER)/%.o : $(SOURCE_FOLDER)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(NVCC_OBJECTS): $(OUTPUT_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cu
	$(NVCC) --compiler-options "$(CFLAGS)" -dc $< -o $@

.PHONY: mkdir_output
mkdir_output:
	@mkdir -p $(OUTPUT_SUBFOLDERS)

.PHONY: clean
clean: clean_app clean_lib clean_interface

.PHONY: clean_app
clean_app:
	@rm -rf $(OUTPUT_FOLDER)

.PHONY: clean_lib
clean_lib:
	@rm -rf lib

.PHONY: clean_interface
clean_interface:
	@$(MAKE) clean -C ./interface
