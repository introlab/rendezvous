LIB_PATH=../../../../../lib
INC_PATH=-I./include -I$(LIB_PATH)/glfw-3.2.1/include -I$(LIB_PATH)/glm/include -I$(LIB_PATH)/glad/include
CC=g++
CFLAGS=-Wall $(INC_PATH)
SOURCE_FOLDERS=$(wildcard src/*)
OUTPUT_FOLDERS=$(patsubst src/%, bin/%, $(SOURCE_FOLDERS))
SOURCES=$(wildcard src/*/*.cpp)
OBJECTS=$(patsubst src/%.cpp, bin/%.o, $(SOURCES))
SOURCES_C=$(wildcard src/*/*.c)
OBJECTS_C=$(patsubst src/%.c, bin/%.o, $(SOURCES_C))
SHADERS_FILES=$(wildcard src/shaders/*.txt)
SHADERS=$(patsubst src/shaders/%.txt, %, $(SHADERS_FILES))

.PHONY: all
all: lib interface

.PHONY: interface
interface: 
	$(MAKE) -C ./interface

.PHONY: lib
lib: clean_shaders build_shaders $(SHADERS) build $(OBJECTS) $(OBJECTS_C) create_lib

$(OBJECTS): bin/%.o : src/%.cpp
	$(CC) $(CFLAGS) -c $< -fPIC -o $@

$(OBJECTS_C): bin/%.o : src/%.c
	$(CC) $(CFLAGS) -c $< -fPIC -o $@

.PHONY: $(SHADERS)
$(SHADERS):
	bash compile_shader.sh $@ src/shaders  src/shaders/autogen

.PHONY: build_shaders
build_shaders:
	@mkdir -p src/shaders/autogen

.PHONY: create_lib
create_lib: 
	@mkdir -p lib
	ar rvs lib/libDewarper.a $(OBJECTS) $(OBJECTS_C)
	
.PHONY: build
build:
	@mkdir -p $(OUTPUT_FOLDERS)

.PHONY: clean
clean: clean_lib clean_python

.PHONY: clean_lib
clean_lib:
	rm -rf *o $(EXECUTABLE) bin lib src/shaders/autogen

.PHONY: clean_python
clean_python:
	$(MAKE) clean -C ./interface

.PHONY: clean_shaders
clean_shaders: 
	rm -rf bin/shaders
