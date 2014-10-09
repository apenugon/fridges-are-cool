all: dicks

dicks: DetectionPipeline.cpp main.cpp
	g++ -std=c++0x -ggdb `pkg-config --cflags opencv` DetectionPipeline.cpp main.cpp -o dicks `pkg-config --libs opencv` -L/usr/local/lib -lopencv_calib3d -lopencv_features2d -lopencv_core -lm
