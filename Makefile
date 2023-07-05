SRC_DIR := src
OBJ_DIR	:= obj
INC_DIR := include

EXE := nnfs
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC))

CXX := g++
CXXVERSION := -std=c++17
CXXFLAGS := -Wall
INCLUDES := -I /usr/local/include/*

.PHONY: all clean

all: $(EXE)

# $(info $$SRC is [${SRC}])
# $(info $$OBJ is [${OBJ}])

$(EXE): $(OBJ)
	$(CXX) $(CXXVERSION) $(INCLUDES) $(CXXFLAGS) $^ -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp | $(OBJ_DIR)
	$(CXX) $(CXXVERSION) $(INCLUDES) $(CXXFLAGS) -c $(SRC_DIR)/main.cpp -o $(OBJ_DIR)/main.o

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp  $(INC_DIR)/%.h | $(OBJ_DIR)
	$(CXX) $(CXXVERSION) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rv $(OBJ_DIR) $(EXE)