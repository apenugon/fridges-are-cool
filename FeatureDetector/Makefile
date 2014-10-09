all: fd

fd: DetectionPipeline.cpp main.cpp
	g++ -std=c++0x -ggdb `pkg-config --cflags opencv` DetectionPipeline.cpp main.cpp `pkg-config --libs opencv` -o fd

clean: fd
	rm fd
