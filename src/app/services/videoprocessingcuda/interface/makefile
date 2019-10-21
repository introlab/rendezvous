
.PHONY: all
all: clean build copy_so

.PHONY: build
build:
	python setup.py build_ext $(NVCC_SOURCES)

.PHONY: copy_so
copy_so:
	cp ./build/*/*.so .

.PHONY: clean
clean:
	@rm -rf build stream.py *.so stream_wrap.cpp
