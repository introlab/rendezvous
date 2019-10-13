.PHONY: all
all: libs dewarping_lib virtualcam_lib

.PHONY: virtualcam_lib
virtualcam_lib:
	$(MAKE) -C ./src/app/services/virtualcameradevice/interface

.PHONY: dewarping_lib
dewarping_lib:
	$(MAKE) -C ./src/app/services/videoprocessing/dewarping

.PHONY: libs
libs:
	$(MAKE) -C ./lib

.PHONY: clean
clean: clean_libs clean_dewarping_lib clean_virtualcam_lib

.PHONY: clean_virtualcam_lib
clean_virtualcam_lib:
	$(MAKE) clean -C ./src/app/services/virtualcameradevice/interface

.PHONY: clean_dewarping_lib
clean_dewarping_lib:
	$(MAKE) clean -C ./src/app/services/videoprocessing/dewarping

.PHONY: clean_libs
clean_libs:
	$(MAKE) clean -C ./lib