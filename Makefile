CXX ?= clang++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Iinclude
LDFLAGS ?= -lpthread

# Termux / Android NDK libc++ often needs explicit c++ runtime
UNAME_S := $(shell uname -s 2>/dev/null || echo unknown)
ifeq ($(shell test -d /data/data/com.termux && echo termux),termux)
  LDFLAGS += -lc++
endif

.PHONY: all clean termux

all: engine

engine: src/engine.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) -o engine src/engine.cpp $(LDFLAGS)

# Explicit Termux recipe (use if plain `make` still fails to link)
termux:
	$(CXX) -O3 -std=c++17 -Wall -Iinclude -o engine src/engine.cpp -lc++ -lpthread
	@echo "Built ./engine for Termux"

clean:
	rm -f engine
