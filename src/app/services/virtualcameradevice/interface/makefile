
.PHONY: all
all: clean build copy_so

.PHONY: build
build:
	python setup.py build_ext

.PHONY: copy_so
copy_so:
	cp ./build/*/*.so .

.PHONY: clean
clean:
	rm -rf build virtual_camera_device.py virtual_camera_device_wrap.cpp *.so