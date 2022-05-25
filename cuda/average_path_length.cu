
#include <climits>
#include <iomanip>
#include <cuda_runtime.h>
#include <queue>
#include "common.h"
#include "kernels.cu"


//__global__ void
//compute_distances_kernel(int *graph, int size, int _x, int _y) {
//    unsigned int x = _x; // column
//    unsigned int y = _y; // row
//    if (_x < 0) {
//        x = threadIdx.x + blockIdx.x * blockDim.x;
//    }
//    if (_y < 0) {
//        y = threadIdx.y + blockIdx.y * blockDim.y;
//    }
//    unsigned int id = y * size + x;
//
//    if (x > size || y > size) {
//        printf(" X (%d,%d)\n", x, y);
//        return;
//    }
//    if (x >= y) {
//        printf("X (%d,%d)\n", x, y);
//        return;
//    } // work on lower half of matrix (upper half is for adjacency)
//
//    printf("(%d,%d) = %d\n", x, y, graph[id]);
//    int min = INT_MAX;
//    for (int node_index = 0; node_index < size; node_index++) {
//        if (graph[id] != 0) return;
//        unsigned int adj_id, dist_id;
//        bool adjacent;
//        if (node_index < y) {
//            adj_id = node_index * size + y;
//            dist_id = y * size + node_index;
//        } else {
//            adj_id = y * size + node_index;
//            dist_id = node_index * size + x;
//        }
//        adjacent = (graph[adj_id] == 1);
//        if (adjacent) {
//            if (graph[dist_id] == 0 && graph[id] == 0) {
//                if (node_index < y) {
//                    printf("  (%d,%d) waiting for (%d,%d)...\n", x, y, node_index, y);
//                    compute_distances_kernel<<<1, 1>>>(graph, size, node_index, y);
//                } else {
//                    printf("  (%d,%d) waiting for (%d,%d)...\n", x, y, x, node_index);
//                    compute_distances_kernel<<<1, 1>>>(graph, size, x, node_index);
//                }
//            }
//            while (graph[dist_id] == 0 && graph[id] == 0); // wait for dependency to clear
//            if (graph[dist_id] < min) {
//                min = graph[dist_id];
//            }
//        }
//    }
//    if (graph[id] == 0) {
//        graph[id] = min + 1;
//        printf("(%d,%d) => %d\n", x, y, graph[id]);
//    }
//}


unsigned int dist_id_of(unsigned int x, unsigned int y, int size) {
    if (x < y) {
        return y * size + x;
    } else {
        return x * size + y;
    }
}

unsigned int adj_id_of(unsigned int x, unsigned int y, int size) {
    if (x < y) {
        return x * size + y;
    } else {
        return y * size + x;
    }
}

void
host_average_dist(int *graph, int size, int source) {
    std::queue<unsigned int> queue;
    queue.push(source);
    bool *visited = new bool[size];
    for (int i = 0; i < size; i++) {
        visited[i] = false;
    }
    visited[source] = true;
    while (!queue.empty()) {
        unsigned int u = queue.front();
        queue.pop();
        printf(" (%d) %d\n", source, u);
        for (int v = 0; v < size; v++) {
            printf(" (%d) - %d\n", source, v);
            unsigned int adj_id = adj_id_of(u, v, size);
            unsigned int dist_id = dist_id_of(source, v, size);
            bool adjacent = (graph[adj_id] == 1);
            if (adjacent) {
                printf("  (%d) %d: %d\n", source, v, graph[dist_id]);
                if (!visited[v]) {
                    graph[dist_id] = graph[dist_id_of(source, u, size)] + 1;
                    visited[v] = true;
                    queue.push(v);
                    printf("(%d) ~> %d -> %d in %d @ %d\n", source, u, v, graph[dist_id], dist_id);
                }
            }
        }
    }
}


void
compute_distances(int size_V, int size_E, const int *adjacency_list, const int *edges_offsets,
                  const int *number_of_edges, int *distances) {
    size_t bytes_V = size_V * sizeof(int);
    size_t bytes_E = size_E * 2 * sizeof(int);
    size_t bytes_distances = size_V * size_V * sizeof(int);

    for (int i = 0; i < size_V; i++) {
        for (int j = 0; j < size_V; j++) {
            distances[i * size_V + j] = INT_MAX;
            if (i == j) {
                distances[i * size_V + j] = 0;
            }
        }
    }

    int *changed;
    CHECK(cudaMallocHost((void **) &changed, sizeof(int)))
    int *_distances;
    CHECK(cudaMalloc((void **) &_distances, bytes_distances))
    CHECK(cudaMemcpy(_distances, distances, bytes_distances, cudaMemcpyHostToDevice))
    int *_adjacency_list;
    CHECK(cudaMalloc((void **) &_adjacency_list, bytes_E))
    CHECK(cudaMemcpy(_adjacency_list, adjacency_list, bytes_E, cudaMemcpyHostToDevice))
    int *_edges_offsets;
    CHECK(cudaMalloc((void **) &_edges_offsets, bytes_V))
    CHECK(cudaMemcpy(_edges_offsets, edges_offsets, bytes_V, cudaMemcpyHostToDevice))
    int *_number_of_edges;
    CHECK(cudaMalloc((void **) &_number_of_edges, bytes_V))
    CHECK(cudaMemcpy(_number_of_edges, number_of_edges, bytes_V, cudaMemcpyHostToDevice))

    dim3 block(size_V / 1024 + 1);
    dim3 grid(1024);

    for (int source = 0; source < size_V; source++) {
        printf("source = %d\n", source);
        std::flush(std::cout);
        int level = 0;
        *changed = 1;
        while (*changed) {
            *changed = 0;
            compute_distances_kernel<<<grid, block>>>(source, size_V, level, _adjacency_list, _edges_offsets,
                                                      _number_of_edges,
                                                      _distances, changed);
            CHECK(cudaDeviceSynchronize())
            printf("level: %d changed: %d\n", level, *changed);
            std::flush(std::cout);
            level++;
        }
    }
    printf("done.\n");

    CHECK(cudaMemcpy(distances, _distances, bytes_distances, cudaMemcpyDeviceToHost))
    CHECK(cudaFree(_distances))
    CHECK(cudaFree(_adjacency_list))
    CHECK(cudaFree(_edges_offsets))
    CHECK(cudaFree(_number_of_edges))
}


void reduceSumDeviceArray(int *g_idata, int *g_odata, int size) {
    dim3 block(512, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    if (size < 1 << 9) {
        // Neighbored
        reduceNeighbored<<<grid, block>>>(g_idata, g_odata, size);
    } else if (size < 1 << 11) {
        // Unrolling2
        reduceUnrolling2<<<grid, block>>>(g_idata, g_odata, size);
    } else if (size < 1 << 12) {
        // Unrolling4
        reduceUnrolling4<<<grid, block>>>(g_idata, g_odata, size);
    } else {
        // UnrollWarp8
        reduceUnrollWarps8<<<grid, block>>>(g_idata, g_odata, size);
    }
}


double
average_path_length(int size_V, int size_E, const int *adjacency_list, const int *edges_offsets,
                    const int *number_of_edges, int *distances) {
    size_t bytes_V = size_V * sizeof(int);
    size_t bytes_E = size_E * 2 * sizeof(int);
    size_t bytes_distances = size_V * size_V * sizeof(int);

    for (int i = 0; i < size_V; i++) {
        for (int j = 0; j < size_V; j++) {
            distances[i * size_V + j] = INT_MAX;
            if (i == j) {
                distances[i * size_V + j] = 0;
            }
        }
    }

    int *changed;
    CHECK(cudaMallocHost((void **) &changed, sizeof(int)))
    int *_distances;
    CHECK(cudaMalloc((void **) &_distances, bytes_distances))
    CHECK(cudaMemcpy(_distances, distances, bytes_distances, cudaMemcpyHostToDevice))
    int *_adjacency_list;
    CHECK(cudaMalloc((void **) &_adjacency_list, bytes_E))
    CHECK(cudaMemcpy(_adjacency_list, adjacency_list, bytes_E, cudaMemcpyHostToDevice))
    int *_edges_offsets;
    CHECK(cudaMalloc((void **) &_edges_offsets, bytes_V))
    CHECK(cudaMemcpy(_edges_offsets, edges_offsets, bytes_V, cudaMemcpyHostToDevice))
    int *_number_of_edges;
    CHECK(cudaMalloc((void **) &_number_of_edges, bytes_V))
    CHECK(cudaMemcpy(_number_of_edges, number_of_edges, bytes_V, cudaMemcpyHostToDevice))

    dim3 block(size_V / 1024 + 1);
    dim3 grid(1024);

    for (int source = 0; source < size_V; source++) {
        printf("source = %d\n", source);
        std::flush(std::cout);
        int level = 0;
        *changed = 1;
        while (*changed) {
            *changed = 0;
            compute_distances_kernel<<<grid, block>>>(source, size_V, level, _adjacency_list, _edges_offsets,
                                                      _number_of_edges,
                                                      _distances, changed);
            CHECK(cudaDeviceSynchronize())
            printf("level: %d changed: %d\n", level, *changed);
            std::flush(std::cout);
            level++;
        }
    }
    printf("done.\n");

//    dim3 cblock(32, 32);
//    dim3 cgrid(size_V, size_V);
//    clean_distances_matrix<<<cgrid, cblock>>>(_distances, size_V);
    CHECK(cudaDeviceSynchronize())
    CHECK(cudaMemcpy(distances, _distances, bytes_distances, cudaMemcpyDeviceToHost))
//    distances[4] = 200;
//    CHECK(cudaMemcpy(_distances, distances, bytes_distances, cudaMemcpyHostToDevice))
    for (int x = 0; x < 10; x++) {
        std::cout << "| ";
        for (int y = 0; y < 10; y++) {
            std::cout << " " << std::setw(2) << std::setfill('0') << distances[x * size_V + y];
        }
        std::cout << " |\n";
    }

    int result_len = (size_V * size_V + 512 - 1) / 512;
    printf("result_len = %d\n", result_len);
    int *result = new int[result_len];
    int *_result;
    int result_bytes = result_len * sizeof(int);
    CHECK(cudaMalloc((void **) &_result, result_bytes))
    CHECK(cudaMemset(_result, 0, result_bytes))
    CHECK(cudaDeviceSynchronize())

    reduceSumDeviceArray(_distances, _result, size_V * size_V);
    CHECK(cudaDeviceSynchronize())
    CHECK(cudaMemcpy(result, _result, result_bytes, cudaMemcpyDeviceToHost))

    int sum = 0;
    for (int i = 0; i < result_len; i++) {
        sum += result[i];
        printf("+= %d\n", result[i]);
    }

    // the resulting sum here will be different from the expected one
    // this is because the size of the array (i.e. distances matrix)
    // is not a power of 2.

//    CHECK(cudaMemcpy(distances, _distances, bytes_distances, cudaMemcpyDeviceToHost))
    CHECK(cudaFree(_result))
    CHECK(cudaFree(_distances))
    CHECK(cudaFree(_adjacency_list))
    CHECK(cudaFree(_edges_offsets))
    CHECK(cudaFree(_number_of_edges))

    return ((double)sum) / (size_V * size_V - size_V);
}
