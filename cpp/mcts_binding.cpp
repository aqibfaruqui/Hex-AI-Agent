#include <torch/extension.h>
#include "BoardState.cpp"
#include "MCTS.cpp"

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS Accelerator for Hex";

    py::class_<BoardState>(m, "BoardState")
        .def(py::init<int>())
        .def("reset", &BoardState::reset)
        .def("play_move", &BoardState::play_move)
        .def("get_legal_moves", &BoardState::get_legal_moves)
        .def("get_result", &BoardState::get_result)
        .def("is_full", &BoardState::is_full)
        .def_readwrite("current_colour", &BoardState::current_colour)
        .def_readwrite("winner", &BoardState::winner)
        .def_readwrite("board", &BoardState::board);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<BoardState, std::string, bool, float>(), 
             py::arg("board"), py::arg("model_path"), py::arg("use_gpu"), py::arg("c_puct")=1.0)
        .def("search", &MCTS::search, py::arg("time_limit"))
        .def("get_action_probs", &MCTS::get_action_probs)
        .def("update_root", &MCTS::update_root);
}