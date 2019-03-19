
.PHONY: all
all: clean build copy_so

.PHONY: build
build:
	python3 setup.py build_ext

.PHONY: copy_so
copy_so:
	cp ./build/*/*.so .

.PHONY: clean
clean:
	rm -rf build fisheye_dewarping.py *.so fisheye_dewarping_wrap.cpp
