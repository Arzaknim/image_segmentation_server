CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pthread

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
INC_DIRS := -I$(SCRIPT_DIR)/include -I$(SCRIPT_DIR)/third_party/onnxruntime/include
LIB_DIRS := -L$(SCRIPT_DIR)/third_party/onnxruntime/lib
RPATH := -Wl,-rpath,'$$ORIGIN/third_party/onnxruntime/lib' -Wl,-rpath,$(SCRIPT_DIR)/third_party/onnxruntime/lib
LIBS := -lonnxruntime -lpthread -ldl -lm

TARGET := dubai_segmentation_cpp

SRCS := $(SCRIPT_DIR)/src/onnx_model.cpp \
        $(SCRIPT_DIR)/src/inference.cpp \
        $(SCRIPT_DIR)/src/logging.cpp \
        $(SCRIPT_DIR)/src/main.cpp

OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LIB_DIRS) $(RPATH) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC_DIRS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

