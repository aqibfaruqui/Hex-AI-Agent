#include "MCTS.h"
#include <chrono>
#include <iostream>
#include <torch/script.h>

torch::Tensor encode_board(const BoardState& b, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor t = torch::zeros({1, 3, 11, 11}, options);
    auto acc = t.accessor<float, 4>();

    int current = b.current_colour;
    int opponent = (current == RED) ? BLUE : RED;

    for(int i = 0; i < b.board_len; i++) {
        int x = i / b.size;
        int y = i % b.size;
        int cell_value = b.board[i];

        if (cell_value == current) {
            // Channel 0: current player's stones
            acc[0][0][x][y] = 1.0f;
        } 
        else if (cell_value == opponent) {
            // Channel 1: opponent's stones
            acc[0][1][x][y] = 1.0f;
        } 
        else {
            // Channel 2: empty squares
            acc[0][2][x][y] = 1.0f;
        }
    }

    return t.to(device);
}

Node::Node(BoardState b, Node* p, int m, float prior_prob) 
    : state(b), parent(p), move(m), prior(prior_prob), visits(0), total_value(0) {}

bool Node::is_leaf() const { return children.empty(); }

float Node::value() const { return (visits == 0) ? 0 : total_value / visits; }

float Node::puct(float c_puct, int parent_visits) const {
    float q = value();
    float u = c_puct * prior * std::sqrt((float)parent_visits) / (1 + visits);
    return q + u;
}

MCTS::MCTS(BoardState root_board, std::string model_path, bool use_gpu, float c) 
    : c_puct(c), device(use_gpu ? torch::kMPS : torch::kCPU) {
    
    if (torch::cuda::is_available()) device = torch::kCUDA;
    
    try {
        model = torch::jit::load(model_path);
        model.to(device);
        model.eval();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Failed to load model: " + e.msg());
    }

    root = std::make_unique<Node>(root_board, nullptr, -1, 1.0f);
}

void MCTS::_run_search(float time_limit) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int simulations = 0;

    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = now - start_time;
        if (elapsed.count() >= time_limit) break;

        simulations++;
        Node* node = root.get();

        // 1: Selection
        while (!node->is_leaf()) {
            float best_score = -1e9;
            Node* best_child = nullptr;
            
            for (auto& child : node->children) {
                float score = child->puct(c_puct, node->visits);
                if (score > best_score) {
                    best_score = score;
                    best_child = child.get();
                }
            }
            if (best_child == nullptr) break;
            node = best_child;
        }

        // 2: Expansion
        float value = 0;
        auto result = node->state.get_result();
        
        if (result.second) { // Game Over
            value = (float)result.first; 
            // If result is +1 (winner is current), and node is current, then previous player (parent) lost
            // Value perspective flipping handled in backpropagation
        } else { // Neural Net Inference
            torch::Tensor input = encode_board(node->state, device);
            std::vector<torch::jit::IValue> inputs = {input};
            
            auto output = model.forward(inputs).toTuple();
            torch::Tensor raw_logits = output->elements()[0].toTensor();    // Reshape output from [1, 11, 11] to [1, 121]
            torch::Tensor flat_logits = raw_logits.reshape({1, -1}); 
            torch::Tensor policy = torch::softmax(flat_logits, 1).cpu();
            value = output->elements()[1].toTensor().item<float>();

            auto policy_acc = policy.accessor<float, 2>();
            for (int move : node->state.get_legal_moves()) {
                BoardState next_state = node->state;
                next_state.play_move(move);
                
                float prob = policy_acc[0][move];
                node->children.push_back(std::make_unique<Node>(next_state, node, move, prob));
            }
        }

        // 3: Backpropagation
        while (node != nullptr) {
            node->visits++;
            node->total_value += value;
            value = -value;
            node = node->parent;
        }
    }
}

int MCTS::search(float time_limit) {
    _run_search(time_limit);
    
    // Return best move (most visits)
    int best_visits = -1;
    int best_move = -1;

    for (auto& child : root->children) {
        if (child->visits > best_visits) {
            best_visits = child->visits;
            best_move = child->move;
        }
    }

    return best_move;
}

std::vector<float> MCTS::get_action_probs() {
    std::vector<float> probs(121, 0.0f);
    int sum_visits = 0;
    for (auto& child : root->children) sum_visits += child->visits;
    
    if (sum_visits > 0) {
        for (auto& child : root->children) {
            probs[child->move] = (float)child->visits / sum_visits;
        }
    }

    return probs;
}

void MCTS::update_root(int move) {
    bool found = false;
    for (auto& child : root->children) {
        if (child->move == move) {
            root = std::move(child);
            root->parent = nullptr; 
            found = true;
            break;
        }
    }
    
    if (!found) {
        BoardState next_state = root->state;
        next_state.play_move(move);
        root = std::make_unique<Node>(next_state, nullptr, -1, 1.0f);
    }
}