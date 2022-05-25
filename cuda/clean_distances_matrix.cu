extern "C" {
__global__ void
clean_distances_matrix(int *distances, int size) {
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx > size || ty > size) return;

    if (tx >= ty) {
        distances[tx * size + ty] = 0;
    }
}
}