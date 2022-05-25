
#ifndef _AVERAGE_PATH_LENGTH_H
#define _AVERAGE_PATH_LENGTH_H

void compute_distances_kernel(int size, int level, const int *adjacency_list, const int *edges_offsets,
                         const int *number_of_edges, int *distances, int *changed);
void compute_distances(int source, int size_V, const int *adjacency_list, const int *edges_offsets,
                       const int *number_of_edges, int *distances);
void average_path_length(const int **graph, int size, double *result);

#endif // _AVERAGE_PATH_LENGTH_H
