
#include <iomanip>
#include <cuda_runtime.h>
#include "average_path_length.cu"
#include "common.h"


using namespace std;


int adj_graph[12 * 2] = {
        3, 4,
        2,
        1, 5,
        0, 4,
        0, 3, 6, 7, 9,
        2, 9,
        4, 7,
        4, 6, 8,
        7, 9,
        4, 5, 8,
};

int edges_offsets[10] = {0, 2, 3, 5, 7, 12, 14, 16, 19, 21};
int number_of_edges[10] = {2, 1, 2, 2, 5, 2, 2, 3, 2, 3};

int expected_distances[10][10] = {
        {00, 05, 04, 01, 01, 03, 02, 02, 03, 02,},
        {05, 00, 01, 05, 04, 02, 05, 05, 04, 03,},
        {04, 01, 00, 04, 03, 01, 04, 04, 03, 02,},
        {01, 05, 04, 00, 01, 03, 02, 02, 03, 02,},
        {01, 04, 03, 01, 00, 02, 01, 01, 02, 01,},
        {03, 02, 01, 03, 02, 00, 03, 03, 02, 01,},
        {02, 05, 04, 02, 01, 03, 00, 01, 02, 02,},
        {02, 05, 04, 02, 01, 03, 01, 00, 01, 02,},
        {03, 04, 03, 03, 02, 02, 02, 01, 00, 01,},
        {02, 03, 02, 02, 01, 01, 02, 02, 01, 00,},
};

#define EPSILON 0.0000000001
bool double_eq(double a, double b)
{
    return fabs(a - b) < EPSILON;
}


bool test_distance_matrix_from_full_adjacency_matrix() {
    int distances[10][10] = {};

    compute_distances(10, 12, adj_graph, edges_offsets, number_of_edges, (int *) distances);
    CHECK(cudaDeviceReset())
    bool correct = true;
    for (int x = 0; x < 10; x++) {
        cout << "| ";
        for (int y = 0; y < 10; y++) {
            cout << " " << setw(2) << setfill('0') << distances[x][y];
            if (distances[x][y] != expected_distances[x][y]) {
                correct = false;
            }
        }
        cout << " |\n";
    }
    return correct;
}

bool test_device_average_path_length() {
    int distances[10][10];
    double l = average_path_length(10, 12, adj_graph, edges_offsets, number_of_edges, (int *) distances);
    CHECK(cudaDeviceReset())

    bool correct_distances = true;
    for (int x = 0; x < 10; x++) {
        cout << "| ";
        for (int y = 0; y < 10; y++) {
            cout << " " << setw(2) << setfill('0') << distances[x][y];
            if (distances[x][y] != expected_distances[x][y]) {
                correct_distances = false;
            }
        }
        cout << " |\n";
    }

    double expected = 2.4666666667;
    cout << "Expected: " << expected << endl;
    cout << "Got:      " << l << endl;
    return double_eq(l, expected) && correct_distances;
}

bool test_host_distance_matrix_from_full_adjacency_matrix() {
    int graph[10][10] = {
            {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 1, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };
    int expected[10][10] = {
            {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
            {5, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {4, 1, 0, 0, 0, 1, 0, 0, 0, 0},
            {1, 5, 4, 0, 1, 0, 0, 0, 0, 0},
            {1, 4, 3, 1, 0, 0, 1, 1, 0, 1},
            {3, 2, 1, 3, 2, 0, 0, 0, 0, 1},
            {2, 5, 4, 2, 1, 3, 0, 1, 0, 0},
            {2, 5, 4, 2, 1, 3, 1, 0, 1, 0},
            {3, 4, 3, 3, 2, 2, 2, 1, 0, 1},
            {2, 3, 2, 2, 1, 1, 2, 2, 1, 0},
    };
//    for (int x = 0; x < 10; x++) {
//        for (int y = 0; y < 10; y++) {
//            host_average_dist((int *) graph, 10, x, y);
//        }
//    }
    for (int x = 0; x < 10; x++) {
        host_average_dist((int *) graph, 10, x);
        for (auto &_x: graph) {
            cout << "| ";
            for (int _y: _x) {
                cout << " " << setw(2) << setfill('0') << _y;
            }
            cout << " |\n";
        }
    }
//    host_average_dist((int*) graph, 10, 0, 9);
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 10; y++) {
            if (graph[x][y] != expected[x][y]) {
                cout << "Got wrong result in test_host_average_dist:\n";
                for (auto &_x: graph) {
                    cout << "| ";
                    for (int _y: _x) {
                        cout << " " << setw(2) << setfill('0') << _y;
                    }
                    cout << " |\n";
                }
                return false;
            }
        }
    }
    return true;
}


int main(int argc, char *argv[]) {
    CHECK(cudaDeviceReset())
    if (argc < 2) {
        cerr << "need test name\n";
        return 1;
    } else {
        if (string(argv[1]) == "nop") {
            return 0;
        } else if (string(argv[1]) == "host") {
            return !test_host_distance_matrix_from_full_adjacency_matrix();
        } else if (string(argv[1]) == "device") {
            return !test_distance_matrix_from_full_adjacency_matrix();
        } else if (string(argv[1]) == "device-path-length") {
            return !test_device_average_path_length();
        } else {
            cerr << "unknown test name\n";
            return 1;
        }
    }
}
