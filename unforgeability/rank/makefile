CXX := g++
CXXFLAGS := -Wall  -std=c++17 -fopenmp -O3

SRC_DIR = src
INCLUDE_DIR = include

EXE1 = experiment-lenet
EXE2 = experiment-resnet
EXE3 = experiment-approx
EXE4 = experiment-vgg

# Source files for each executable
EXE1_SRCS = $(SRC_DIR)/lenet.cpp $(SRC_DIR)/gaussian.cpp $(SRC_DIR)/read-files.cpp
EXE2_SRCS = $(SRC_DIR)/resnet.cpp $(SRC_DIR)/gaussian.cpp $(SRC_DIR)/read-files.cpp
EXE3_SRCS = $(SRC_DIR)/approx.cpp $(SRC_DIR)/gaussian.cpp $(SRC_DIR)/read-files.cpp
EXE4_SRCS = $(SRC_DIR)/vgg.cpp $(SRC_DIR)/gaussian.cpp $(SRC_DIR)/read-files.cpp

DIR_PATH1 ?= ./data-approx/
DIR_PATH2 ?= ./data-lenet/
DIR_PATH3 ?= ./data-resnet/
DIR_PATH4 ?= ./data-vgg/
# Create executables
all: $(EXE1) $(EXE2) $(EXE3) $(EXE4)

$(EXE1): $(EXE1_SRCS)
	$(CXX) $(CXXFLAGS) -DDIR_PATH='"$(DIR_PATH1)"' -o $@ $^

$(EXE2): $(EXE2_SRCS)
	$(CXX) $(CXXFLAGS) -DDIR_PATH='"$(DIR_PATH2)"' -o $@ $^

$(EXE3): $(EXE3_SRCS)
	$(CXX) $(CXXFLAGS) -DDIR_PATH='"$(DIR_PATH3)"' -o $@ $^

$(EXE4): $(EXE4_SRCS)
	$(CXX) $(CXXFLAGS) -DDIR_PATH='"$(DIR_PATH4)"' -o $@ $^
# Clean
.PHONY: clean
clean:
	rm -f $(EXE1) $(EXE2) $(EXE3) $(EXE4)

