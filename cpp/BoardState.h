#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <memory>

enum Colour { EMPTY = 0, RED = 1, BLUE = 2 };

class BoardState {
public:
    int size;
    int board_len;
    std::vector<int8_t> board; 
    int current_colour;        
    int winner;                
    int moves_count;

    std::vector<int16_t> parent;
    std::vector<int8_t> rank;
    int RED_TOP, RED_BOTTOM;
    int BLUE_LEFT, BLUE_RIGHT;

    BoardState(int sz = 11);

    int find(int x);
    void union_sets(int a, int b);

    void reset();
    std::vector<int> get_legal_moves() const;
    void play_move(int cell);
    std::pair<int, bool> get_result() const; // Returns (value, finished)
    bool is_full() const;
};