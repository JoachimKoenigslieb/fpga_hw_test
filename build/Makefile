SRC_DIR := ../src
KERNEL_DIR := ${SRC_DIR}/kernels
OBJ_DIR := ./objects

TARGET := hw
PLATFORM := 

KERNEL_INCLUDES := -I/tools/Xilinx/Vivado/2020.1/include/

HOST_INCLUDES := -I/opt/xilinx/xrt/include/ -I/tools/Xilinx/Vivado/2020.1/include/ -I${SRC_DIR}/includes/

HOST_LIBS  := -L/opt/xilinx/xrt/lib/ -L/usr/local/lib -L/usr/lib -lOpenCL -lz -lpthread -lrt -lstdc++ -std=c++14 -w -O0 -g 

SOURCES := $(wildcard ${KERNEL_DIR}/*.cpp)
OBJECTS := $(patsubst %.cpp,%.xo,$(addprefix $(TARGET)_objects/,$(notdir $(SOURCES))))

all: host ${TARGET}_kernels.xclbin

$(TARGET)_kernels.xclbin: $(OBJECTS) 
	emconfigutil --platform ${PLATFORM}
	v++ -t $(TARGET) --platform ${PLATFORM} --config design.cfg -l -g -o $@ $^

$(TARGET)_objects/%.xo: $(KERNEL_DIR)/%.cpp
	v++ -t $(TARGET) --platform ${PLATFORM} --config design.cfg -c -k $(basename $(notdir $<)) $(KERNEL_INCLUDES) -o $@ $<

emconfig.json:
	emconfigutil --platform ${PLATFORM}

host: $(SRC_DIR)/host.cpp
	g++ -o host $< ${HOST_INCLUDES} ${HOST_LIBS} -g

clean:
	rm -f *.log
	rm -f *_summary
	rm -f *.info
	rm -rf ./_x
	rm -rf ./.run
	rm -rf ./.Xil
	rm ./objects_hw_emu/*compile_summary
	rm ./objects_sw_emu/*compile_summary
