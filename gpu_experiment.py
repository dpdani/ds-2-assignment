import itertools
from pathlib import Path

import cupy
import igraph
import numpy as np
from cupyx import jit

from experiment import Model, Protocol
from graph import Graph


distances_source = Path().cwd() / "cuda" / "distances_kernel.cu"
# clean_distances_matrix_source = Path().cwd() / "cuda" / "clean_distances_matrix.cu"

with open(distances_source, 'r') as f:
    compute_distances_kernel = cupy.RawKernel(
        f.read(), "compute_distances_kernel", backend='nvcc'
    )

# with open(clean_distances_matrix_source, 'r') as f:
#     clean_distances_matrix = cupy.RawKernel(
#         f.read(), "clean_distances_matrix", backend='nvcc'
#     )

# compute_distances_kernel = kernels_module.get_function("compute_distances_kernel")
# clean_distances_matrix = kernels_module.get_function("clean_distances_matrix")


DTYPE = 'int32'


class GpuModel(Model):
    def __init__(
            self,
            proto: Protocol,
            graph: Graph,
            nodes: int,
            view_size: int,
            view_to_send_size: int,
            delta_t: int,
            disaster_intensity: float,
    ):
        super().__init__(
            proto,
            graph,
            nodes,
            view_size,
            view_to_send_size,
            delta_t,
            disaster_intensity,
        )

    def average_path_length(self):
        return self.gpu_average_path_length(self.overlay_network)

    def alive_average_path_length(self):
        return self.gpu_average_path_length(self.alive_overlay_network)

    @staticmethod
    def gpu_average_path_length(graph: igraph.Graph):
        if graph.ecount() == 0:
            return 0.0

        def prepare():
            adj_list = graph.get_adjlist()
            # print(f"{adj_list=}")
            size_V = graph.vcount()
            # print(f"{size_V=}")
            edges_offsets = cupy.ndarray((size_V,), dtype=DTYPE)
            offset = 0
            number_of_edges = cupy.ndarray((size_V,), dtype=DTYPE)
            for node in range(size_V):
                edges_offsets[node] = offset
                edges = len(adj_list[node])
                number_of_edges[node] = edges
                offset += edges
            distances = cupy.full((size_V, size_V), 2147483647, dtype=DTYPE)
            cupy.fill_diagonal(distances, 0)
            distances = cupy.squeeze(cupy.asarray(distances))
            adj_list = cupy.array(list(
                itertools.chain.from_iterable(adj_list)  # flatten
            ), dtype=DTYPE)
            return adj_list, size_V, edges_offsets, number_of_edges, distances

        adj_list, size_V, edges_offsets, number_of_edges, distances = prepare()

        # print(f"{adj_list=}")
        # print(f"{edges_offsets=}")
        # print(f"{number_of_edges=}")
        # print(f"{distances=}")

        def compute(source, size_V, level, adj_list, edges_offsets, number_of_edges, distances, changed):
            compute_distances_kernel(
                (1024,), (size_V // 1024 + 1,),
                (source, size_V, level, adj_list, edges_offsets, number_of_edges, distances, changed,)
            )
            cupy.cuda.runtime.deviceSynchronize()

        source: igraph.Vertex
        for source in graph.vs:
            level = 0
            changed = cupy.array([1, 1])
            # print(f"{(source.index, size_V, level, adj_list, edges_offsets, number_of_edges, distances, changed,)=}")
            while changed[0]:
                changed[0] = 0
                compute(source.index, size_V, level, adj_list, edges_offsets, number_of_edges, distances, changed,)
                level += 1
                # print(f"{level=}")
                # print(f"{changed=}")
        # print(f"{distances=}")
        # clean_distances_matrix((32, 32), (size_V, size_V), (distances, size_V,))
        # cupy.cuda.runtime.deviceSynchronize()
        s = cupy.sum(distances).item()
        # print(f"{s=}")
        return s / (size_V * size_V - size_V)

    @staticmethod
    def gpu_average_path_length_py(graph: igraph.Graph):

        @jit.rawkernel()  # the following is actually C++ code
        def average_path_length(graph, result):
            for source in range(0, len(graph)):
                pass

        adjacency_matrix = cupy.array(graph.get_adjacency(igraph.GET_ADJACENCY_UPPER), dtype=bool)
        result: float = -1.0
        average_path_length[16, 16](adjacency_matrix, result)

        return result
