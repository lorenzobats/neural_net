CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
SRC = src/*.cpp
INCLUDE = -Iinclude
TARGET = ./build/neural_net

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
