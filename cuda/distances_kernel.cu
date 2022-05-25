extern "C" {
__global__ void
compute_distances_kernel(int source, int size, int level, const int *adjacency_list, const int *edges_offsets,
                         const int *number_of_edges, int *distances, int *changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int _changed = 0;

    if (tid < size && distances[source * size + tid] == level) {
        int u = tid;
//        printf("%d: %d x%d\n", u, edges_offsets[u], number_of_edges[u]);
        for (int i = edges_offsets[u]; i < edges_offsets[u] + number_of_edges[u]; i++) {
            int v = adjacency_list[i];
//            printf("%d -> %d/%d (%d < %d)\n", u, v, i, level + 1, distances[source * size + v]);
            if (level + 1 < distances[source * size + v]) {
                distances[source * size + v] = level + 1;
//                printf("%d(%d,%d) => %d\n", source, u, v, distances[source * size + v]);
                _changed = 1;
            }
        }
    }

    if (_changed) {
        *changed = 1;
    }
}
}