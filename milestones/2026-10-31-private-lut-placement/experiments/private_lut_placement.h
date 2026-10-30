#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Constructive placement component only; no BFV/OPRF security certification.
namespace private_lut {
using Triple = std::array<std::size_t, 3>;

inline Triple balanced_triple(__uint128_t tag, std::size_t slots) {
    if (slots < 3 || slots > (1ULL << 20)) throw std::invalid_argument("slots");
    const __uint128_t d = (__uint128_t)slots * (slots - 1) * (slots - 2);
    auto rank = tag % d;
    Triple v{};
    v[0] = rank % slots; rank /= slots;
    v[1] = rank % (slots - 1); rank /= slots - 1;
    if (v[1] >= v[0]) ++v[1];
    v[2] = rank;
    const auto lo = v[0] < v[1] ? v[0] : v[1];
    const auto hi = v[0] < v[1] ? v[1] : v[0];
    if (v[2] >= lo) ++v[2];
    if (v[2] >= hi) ++v[2];
    return v;
}

inline __uint128_t little_endian_tag(const unsigned char* p) {
    __uint128_t value = 0;
    for (unsigned i = 0; i < 16; ++i) value |= (__uint128_t)p[i] << (8 * i);
    return value;
}

class Matching {
public:
    static constexpr auto none = std::numeric_limits<std::size_t>::max();
    explicit Matching(std::size_t slots) : owner_(slots, none) {}

    // All existing vertices are matched. BFS explores every alternating path.
    // A failed insertion leaves the previous graph and matching unchanged.
    bool insert(const Triple& neighbors) {
        for (auto s : neighbors) if (s >= owner_.size()) throw std::invalid_argument("neighbor");
        if (neighbors[0] == neighbors[1] || neighbors[0] == neighbors[2] ||
            neighbors[1] == neighbors[2]) throw std::invalid_argument("duplicate neighbor");
        const auto root = graph_.size();
        graph_.push_back(neighbors); assigned_.push_back(none);
        seen_.push_back(0); parent_.push_back(none);
        if (++epoch_ == 0) throw std::overflow_error("matching epoch");
        queue_.clear(); queue_.push_back(root); seen_[root] = epoch_;
        last_moves_ = 0;
        for (std::size_t head = 0; head < queue_.size(); ++head) {
            const auto key = queue_[head];
            for (auto slot : graph_[key]) {
                const auto occupied = owner_[slot];
                if (occupied == none) {
                    auto move_key = key, move_slot = slot;
                    while (true) {
                        const auto old_slot = assigned_[move_key];
                        owner_[move_slot] = move_key;
                        assigned_[move_key] = move_slot;
                        ++last_moves_;
                        if (move_key == root) return true;
                        move_slot = old_slot; move_key = parent_[move_key];
                    }
                }
                if (seen_[occupied] != epoch_) {
                    seen_[occupied] = epoch_; parent_[occupied] = key;
                    queue_.push_back(occupied);
                }
            }
        }
        graph_.pop_back(); assigned_.pop_back(); seen_.pop_back(); parent_.pop_back();
        return false;
    }

    bool valid() const {
        std::size_t count = 0;
        for (std::size_t s = 0; s < owner_.size(); ++s) {
            const auto k = owner_[s];
            if (k == none) continue;
            if (k >= graph_.size() || assigned_[k] != s) return false;
            ++count;
        }
        if (count != graph_.size()) return false;
        for (std::size_t k = 0; k < graph_.size(); ++k) {
            auto s = assigned_[k];
            if (s == none || owner_[s] != k) return false;
            if (s != graph_[k][0] && s != graph_[k][1] && s != graph_[k][2]) return false;
        }
        return true;
    }
    const std::vector<std::size_t>& assignment() const { return assigned_; }
    std::size_t last_moves() const { return last_moves_; }
private:
    std::vector<Triple> graph_;
    std::vector<std::size_t> owner_, assigned_, seen_, parent_, queue_;
    std::size_t epoch_ = 0, last_moves_ = 0;
};
} // namespace private_lut
