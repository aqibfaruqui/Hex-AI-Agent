#pragma once
#include "BoardState.h"
#include <memory>
#include <unordered_map>
#include <torch/extension.h>

class Node {
public:
    int move;
    Node* parent;
    std::vector<std::unique_ptr<Node>> children;
    int visits;         // N
    float total_value;  // W
    float prior;        // P
    BoardState state;

    Node(BoardState b, Node* p, int m, float prior_prob);

    bool is_leaf() const;
    float value() const; // Q
    float puct(float c_puct, int parent_visits) const;
};

class MCTS {
public:
    std::unique_ptr<Node> root;
    float c_puct;
    torch::jit::script::Module model;
    torch::Device device;

    MCTS(BoardState root_board, std::string model_path, bool use_gpu, float c = 1.0);

    int search(float time_limit);
    std::vector<float> get_action_probs(); // Returns policy vector
    void update_root(int move);

private:
    void _run_search(float time_limit);
};