.PHONY: lib, pybind, clean, format, all

all: lib


lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; ${MAKE}

compile:
	@mkdir -p build
	@cd build; cmake -S ../
	@cd build; ${MAKE}