all: fern

fern: fern.cpp
	g++ -ggdb `pkg-config --cflags opencv` fern.cpp -o fern `pkg-config --libs opencv`

