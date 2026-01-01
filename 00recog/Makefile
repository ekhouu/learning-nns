CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g

# Default target
all: out/tests

out/tests: tests.cpp network.cpp network.hpp | out
	$(CXX) $(CXXFLAGS) tests.cpp network.cpp -o out/tests

out:
	mkdir -p out

clean:
	rm -rf out *.o

.PHONY: all clean
