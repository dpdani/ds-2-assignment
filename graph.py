import itertools
from enum import Enum

import igraph
import networkx


class Graph(Enum):
    GEO = "geo"
    RANDOM = "random"
    LATTICE = "lattice"
    STAR = "star"

    @staticmethod
    def make_geo(local_nodes: int):
        # 6 global nodes
        geo = networkx.complete_graph(6)
        for edge in geo.edges:
            geo.edges[edge]['latency'] = 10
        # 6 geographical nodes, for each core node
        for n in range(6):
            nodes = len(geo)  # starting nodes
            geo.add_nodes_from([
                _ for _ in range(nodes, nodes + 6)
            ])
            geo.add_edges_from(list(filter(None, [
                (x, y, {'latency': 4}) if x != y else None
                for x in range(nodes, nodes + 6)
                for y in range(nodes, nodes + 6)
            ])))
            geo.add_edges_from([
                (n, x, {'latency': 4})
                for x in range(nodes, nodes + 6)
            ])
        # local nodes
        for i in range(36):
            local = networkx.relabel_nodes(
                networkx.complete_graph(local_nodes // 36),
                {
                    _: _ + len(geo)
                    for _ in range(local_nodes // 36)
                }
            )
            for edge in local.edges:
                local.edges[edge]['latency'] = 1
            geo = networkx.union(geo, local)
            geo.add_edges_from([
                (_, 6 + i, {"latency": 1})
                for _ in local
            ])
        return geo

    # Higher performance version (~5x faster).
    # Made this because the other implementation took ~5min to complete
    # for 10_000 nodes.
    # Also, the performance bottleneck here is the `.to_networkx()` at the end.
    # I regret not choosing this library at the beginning :(
    @staticmethod
    def make_i_geo(local_nodes: int):
        def edges(g: int, from_local: int, to_local: int):
            local = itertools.count(from_local)
            for y in local:
                if y == to_local:
                    break
                yield y, g

        def full_graph(from_: int, to: int):
            X = itertools.count(from_)
            for x in X:
                if x == to:
                    return
                Y = itertools.count(from_)
                for y in Y:
                    if y == to:
                        break
                    if x != y and y < x:
                        yield x, y

        geo: igraph.Graph = igraph.Graph(
            6 +  # 6 global nodes
            36 +  # 6 geographical nodes, for each core node
            (local_nodes // 36) * 36
        )
        geo.add_edges(full_graph(0, 6))
        for i in range(6):
            geo.add_edges(full_graph(
                6 + i * 6,
                6 + i * 6 + 6,
            ))
            geo.add_edges(
                filter(lambda _: _ is not None, [
                    (i, x) if i != x else None
                    for x in range(6 + i * 6, 6 + i * 6 + 6)
                ])
            )
        for i in range(36):
            geo.add_edges(full_graph(
                42 + i * (local_nodes // 36),
                42 + i * (local_nodes // 36) + (local_nodes // 36),
            ))
            geo.add_edges(edges(
                6 + i,
                42 + i * (local_nodes // 36),
                42 + i * (local_nodes // 36) + (local_nodes // 36),
            ))
        geo.es[:]["latency"] = 1
        for edge_i in range(len(geo.es)):
            x, y = (geo.es[edge_i].source, geo.es[edge_i].target)
            if 0 <= x <= 5 and 0 <= y <= 5:
                geo.es[edge_i]["latency"] = 10
            elif 0 <= x <= 41 and 0 <= y <= 41:
                geo.es[edge_i]["latency"] = 4
        return geo

    @staticmethod
    def make_random(nodes: int):
        net = networkx.erdos_renyi_graph(nodes, 0.1)
        for edge in net.edges:
            net.edges[edge]['latency'] = 1
        return net

    @staticmethod
    def make_i_random(nodes: int):
        net = igraph.Graph.Erdos_Renyi(nodes, 0.1)
        net.es[:]["latency"] = 1
        return net

    @staticmethod
    def make_lattice(nodes: int):
        net = networkx.ring_of_cliques(nodes // 2, 2)
        for edge in net.edges:
            net.edges[edge]['latency'] = 1
        return net

    @staticmethod
    def make_i_lattice(nodes: int):
        net = igraph.Graph.Lattice([nodes], circular=True)
        net.es[:]["latency"] = 1
        return net

    @staticmethod
    def make_star(nodes: int):
        net = networkx.star_graph(nodes)
        for edge in net.edges:
            net.edges[edge]['latency'] = 1
        return net

    @staticmethod
    def make_i_star(nodes: int):
        net = igraph.Graph.Star(nodes, center=0)
        net.es[:]["latency"] = 1
        return net

    def make(self, nodes: int) -> networkx.Graph:
        match self:
            case Graph.GEO:
                return self.make_geo(nodes)
            case Graph.RANDOM:
                return self.make_random(nodes)
            case Graph.LATTICE:
                return self.make_lattice(nodes)
            case Graph.STAR:
                return self.make_star(nodes)

    def i_make(self, nodes: int) -> igraph.Graph:
        match self:
            case Graph.GEO:
                return self.make_i_geo(nodes)
            case Graph.RANDOM:
                return self.make_i_random(nodes)
            case Graph.LATTICE:
                return self.make_i_lattice(nodes)
            case Graph.STAR:
                return self.make_i_star(nodes)
