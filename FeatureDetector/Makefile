all: dicks

dicks: DetectionPipeline.cpp main.cpp
	g++ -ggdb `pkg-config --cflags opencv` DetectionPipeline.cpp main.cpp -o dicks `pkg-config --libs opencv`
