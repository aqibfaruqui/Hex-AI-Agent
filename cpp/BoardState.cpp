#include "BoardState.h"

BoardState::BoardState(int sz) : size(sz), board_len(sz * sz) {
    RED_TOP = board_len;
    RED_BOTTOM = board_len + 1;
    BLUE_LEFT = board_len + 2;
    BLUE_RIGHT = board_len + 3;
    
    reset();
}

void BoardState::reset() {
    board.assign(board_len, EMPTY);
    current_colour = RED; 
    winner = EMPTY;
    moves_count = 0;

    parent.resize(board_len + 4);
    rank.assign(board_len + 4, 0);
    std::iota(parent.begin(), parent.end(), 0);
}

int BoardState::find(int x) {
    int root = x;
    while (root != parent[root]) {
        root = parent[root];
    }
    
    while (x != root) {
        int next = parent[x];
        parent[x] = root;
        x = next;
    }
    
    return root;
}

void BoardState::union_sets(int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b) {
        if (rank[a] < rank[b]) {
            parent[a] = b;
        } else if (rank[a] > rank[b]) {
            parent[b] = a;
        } else {
            parent[b] = a;
            rank[a]++;
        }
    }
}

std::pair<int, bool> BoardState::get_result() const {
    if (winner != EMPTY) {
        // +1 if current player won, -1 if lost
        return (winner == current_colour) ? std::make_pair(1, true) : std::make_pair(-1, true);
    }
    return std::make_pair(0, false);
}

std::vector<int> BoardState::get_legal_moves() const {
    std::vector<int> moves;
    moves.reserve(board_len - moves_count);
    for (int i = 0; i < board_len; i++) {
        if (board[i] == EMPTY) moves.push_back(i);
    }
    return moves;
}

void BoardState::play_move(int cell) {
    if (board[cell] != EMPTY || winner != EMPTY) return;

    board[cell] = current_colour;
    moves_count++;

    int x = cell / size;
    int y = cell % size;

    const int dx[] = {-1, -1, 0, 0, 1, 1};
    const int dy[] = {0, 1, -1, 1, -1, 0};

    // Connect to Neighbours
    for (int i = 0; i < 6; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
            int neighbor_idx = nx * size + ny;
            if (board[neighbor_idx] == current_colour) {
                union_sets(cell, neighbor_idx);
            }
        }
    }

    // Connect to Edges
    if (current_colour == RED) {
        if (x == 0) union_sets(cell, RED_TOP);
        if (x == size - 1) union_sets(cell, RED_BOTTOM);
        if (find(RED_TOP) == find(RED_BOTTOM)) winner = RED;
    } else {
        if (y == 0) union_sets(cell, BLUE_LEFT);
        if (y == size - 1) union_sets(cell, BLUE_RIGHT);
        if (find(BLUE_LEFT) == find(BLUE_RIGHT)) winner = BLUE;
    }

    current_colour = (current_colour == RED) ? BLUE : RED;
}

bool BoardState::is_full() const { return moves_count == board_len; }