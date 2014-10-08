all: dicks

dicks: DetectionPipeline.cpp main.cpp
	g++ -std=c++0x -L/usr/local/lib -ggdb `pkg-config --cflags opencv` DetectionPipeline.cpp main.cpp -o dicks `pkg-config --libs opencv`
