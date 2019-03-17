.PHONY: all
all: build_glfw

.PHONY: build_glfw
build_glfw:
	@mkdir -p ./glfw-3.2.1/build
	cmake ./glfw-3.2.1 -B./glfw-3.2.1/build
	$(MAKE) -C ./glfw-3.2.1/build

.PHONY: clean
clean: clean_glfw

.PHONY: clean_glfw
clean_glfw:
	$(MAKE) clean -C ./glfw-3.2.1/build