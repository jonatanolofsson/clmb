#pragma once
#include <vector>
#include <set>
#include <map>

namespace lmb {
    template<typename T>
    struct ConnectedComponents {
        using Connections = std::vector<T*>;
        using CSet = std::set<T*>;
        using CMap = std::map<T*, CSet>;
        CMap connections;

        void init(T* const k) {
            connections.emplace(k, CSet());
        }

        void connect(T* const k, const Connections& cs) {
            connections[k].insert(cs.begin(), cs.end());
        }

        bool get_component(Connections& res) {
            res.clear();
            Connections nodes;
            if (connections.empty()) {
                return false;
            }

            nodes.push_back(connections.begin()->first);
            while(!nodes.empty()) {
                auto k = nodes.back();
                nodes.pop_back();
                res.push_back(k);
                for (const auto& s : res) {
                    connections[k].erase(s);
                }
                for (const auto& s : nodes) {
                    connections[k].erase(s);
                }
                nodes.insert(nodes.end(), connections[k].begin(), connections[k].end());
                connections.erase(k);
            }

            return true;
        }
    };

}
