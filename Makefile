rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

SRC_DIR := src
OBJ_DIR	:= obj
INC_DIR := include

FULL := $(patsubst $(SRC_DIR)/%.cpp, %.cpp, $(call rwildcard, $(SRC_DIR), *.cpp))
FILE := $(notdir $(FULL))
FILE_PATH := $(sort $(dir $(FULL)))
OBJ_SUB_DIR := $(addprefix $(OBJ_DIR)/, $(patsubst %/, %, $(FILE_PATH)))

EXE := nnfs
SRC := $(call rwildcard, $(SRC_DIR), *.cpp)
OBJ := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC))

CXX := g++
CXXVERSION := -std=c++17
CXXFLAGS := -Wall -Ofast -march=native
INCLUDES := -I /usr/local/include/*

.PHONY: all clean

all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(CXXVERSION) $(INCLUDES) $(CXXFLAGS) $^ -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp | $(OBJ_DIR)
	$(CXX) $(CXXVERSION) $(INCLUDES) $(CXXFLAGS) -c $(SRC_DIR)/main.cpp -o $(OBJ_DIR)/main.o

.SECONDEXPANSION:
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp  $(INC_DIR)/%.h | $(OBJ_DIR) $$(@D)
	$(CXX) $(CXXVERSION) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

$(OBJ_SUB_DIR):
	mkdir -p $@

clean:
	rm -rv $(OBJ_DIR) $(EXE)