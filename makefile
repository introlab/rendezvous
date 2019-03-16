.PHONY: all
all: libs dewarping_lib

.PHONY: dewarping_lib
dewarping_lib:
	$(MAKE) -C ./src/app/videoprocessing/dewarping

.PHONY: libs
libs:
	$(MAKE) -C ./lib

.PHONY: clean
clean: clean_libs clean_dewarping_lib 

.PHONY: clean_dewarping_lib
clean_dewarping_lib:
	$(MAKE) clean -C ./src/app/videoprocessing/dewarping

.PHONY: clean_libs
clean_libs:
	$(MAKE) clean -C ./lib