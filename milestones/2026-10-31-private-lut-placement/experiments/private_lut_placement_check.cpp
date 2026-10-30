#include "private_lut_placement.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

using private_lut::Matching;
using private_lut::Triple;
static void check(bool ok) { if (!ok) throw std::runtime_error("placement check failed"); }

// Independent finite Hall oracle: enumerate every left subset, no matching.
bool hall(const std::vector<Triple>& graph, std::size_t prefix) {
    for (std::size_t mask = 1; mask < (1ULL << prefix); ++mask) {
        unsigned neighbors = 0;
        for (std::size_t k = 0; k < prefix; ++k) if (mask & (1ULL << k))
            for (auto s : graph[k]) neighbors |= 1U << s;
        if (__builtin_popcount(neighbors) < __builtin_popcountll(mask)) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 6) throw std::runtime_error("tags.bin mapped.bin metrics.json cases.bin case_maps.bin");
    auto start = std::chrono::steady_clock::now();
    std::uint64_t graphs = 0, prefixes = 0, infeasible = 0, map_ranks = 0;
    for (std::size_t m = 4; m <= 6; ++m) {
        std::vector<Triple> choices;
        for (std::size_t a = 0; a < m; ++a) for (std::size_t b = a + 1; b < m; ++b)
            for (std::size_t c = b + 1; c < m; ++c) choices.push_back({a,b,c});
        for (std::size_t n = 1; n <= (m == 6 ? 4 : 5); ++n) {
            std::uint64_t count = 1;
            for (std::size_t k = 0; k < n; ++k) count *= choices.size();
            for (std::uint64_t i = 0; i < count; ++i) {
                auto code = i; std::vector<Triple> graph;
                for (std::size_t k = 0; k < n; ++k) {
                    graph.push_back(choices[code % choices.size()]); code /= choices.size();
                }
                Matching matching(m);
                for (std::size_t k = 0; k < n; ++k) {
                    auto before = matching.assignment();
                    auto result = matching.insert(graph[k]);
                    check(result == hall(graph, k + 1)); ++prefixes;
                    check(matching.valid());
                    if (!result) { check(matching.assignment() == before); ++infeasible; break; }
                }
                ++graphs;
            }
        }
    }
    for (std::size_t m = 3; m <= 32; ++m) {
        auto d = m * (m - 1) * (m - 2);
        std::set<Triple> seen;
        for (std::size_t rank = 0; rank < d; ++rank) {
            auto v = private_lut::balanced_triple(rank, m);
            check(v[0] < m && v[1] < m && v[2] < m);
            check(v[0] != v[1] && v[0] != v[2] && v[1] != v[2]);
            check(v == private_lut::balanced_triple(rank + d, m));
            check(seen.insert(v).second); ++map_ranks;
        }
        check(seen.size() == d);
    }
    Matching chain(1026);
    check(chain.insert({0,1,2})); check(chain.insert({1,2,0}));
    for (std::size_t i = 2; i <= 1024; ++i) check(chain.insert({i,i+1,0}));
    check(chain.insert({0,1,2})); check(chain.valid());
    const auto chain_moves = chain.last_moves(); check(chain_moves > 500);
    bool bad_range = false, bad_duplicate = false;
    try { chain.insert({0,1,1026}); } catch (const std::invalid_argument&) { bad_range = true; }
    try { chain.insert({0,0,1}); } catch (const std::invalid_argument&) { bad_duplicate = true; }
    check(bad_range && bad_duplicate && chain.valid());

    std::ifstream cases(argv[4], std::ios::binary);
    std::ofstream case_maps(argv[5], std::ios::binary);
    check(bool(cases) && bool(case_maps));
    std::size_t case_count = 0;
    while (true) {
        std::uint32_t slots;
        if (!cases.read(reinterpret_cast<char*>(&slots), 4)) { check(cases.eof()); break; }
        unsigned char bytes[16]; cases.read(reinterpret_cast<char*>(bytes), 16); check(bool(cases));
        auto triple = private_lut::balanced_triple(private_lut::little_endian_tag(bytes), slots);
        for (auto s : triple) {
            auto word = static_cast<std::uint32_t>(s);
            case_maps.write(reinterpret_cast<char*>(&word), 4);
        }
        ++case_count;
    }
    case_maps.close(); check(bool(case_maps));

    constexpr std::size_t N = 262399, B = 2048, M = 4096;
    std::ifstream input(argv[1], std::ios::binary);
    std::ofstream mapped(argv[2], std::ios::binary);
    check(bool(input) && bool(mapped));
    std::vector<Triple> buckets(N), positions(N);
    std::vector<std::vector<std::size_t>> loads(B);
    for (std::size_t id = 0; id < N; ++id) {
        unsigned char tags[32]; input.read(reinterpret_cast<char*>(tags), 32);
        check(bool(input));
        buckets[id] = private_lut::balanced_triple(private_lut::little_endian_tag(tags), B);
        positions[id] = private_lut::balanced_triple(private_lut::little_endian_tag(tags + 16), M);
        for (auto b : buckets[id]) loads[b].push_back(id);
        for (const auto& v : {buckets[id], positions[id]}) for (auto s : v) {
            auto word = static_cast<std::uint32_t>(s);
            mapped.write(reinterpret_cast<char*>(&word), 4);
        }
    }
    char extra; check(!input.read(&extra, 1));
    std::size_t maximum = 0, placed = 0;
    for (const auto& ids : loads) {
        maximum = std::max(maximum, ids.size());
        Matching matching(M);
        for (auto id : ids) check(matching.insert(positions[id]));
        check(matching.valid()); placed += ids.size();
    }
    check(placed == 3*N); check(maximum <= 600);
    Matching query(B);
    for (std::size_t i = 0; i < 256; ++i) check(query.insert(buckets[(i * 1031 + 151935) % N]));
    check(query.valid());
    mapped.close(); check(bool(mapped));
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ofstream metrics(argv[3]);
    metrics << "{\"all_passed\":true,\"graphs\":" << graphs << ",\"prefixes\":" << prefixes
        << ",\"infeasible_prefixes\":" << infeasible << ",\"map_ranks\":" << map_ranks
        << ",\"chain_moves\":" << chain_moves << ",\"boundary_random_cases\":" << case_count << ",\"fullsize_ids\":" << N
        << ",\"server_placements\":" << placed << ",\"max_bucket_load\":" << maximum
        << ",\"server_buckets\":" << B << ",\"query_count\":256,\"elapsed_seconds\":" << elapsed << "}\n";
    check(bool(metrics));
    std::cout << "placement standalone checks passed, graphs=" << graphs << ", maxload=" << maximum << '\n';
}
