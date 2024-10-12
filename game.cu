#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define ROWS 6
#define COLS 7

// Utility function to check if a move leads to a win
__device__ bool check_win(int board[ROWS][COLS], int player) {
    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS; ++col) {
            if (col + 3 < COLS && 
                board[row][col] == player && 
                board[row][col + 1] == player && 
                board[row][col + 2] == player && 
                board[row][col + 3] == player) {
                return true;
            }
            if (row + 3 < ROWS &&
                board[row][col] == player &&
                board[row + 1][col] == player &&
                board[row + 2][col] == player &&
                board[row + 3][col] == player) {
                return true;
            }
            if (row + 3 < ROWS && col + 3 < COLS &&
                board[row][col] == player &&
                board[row + 1][col + 1] == player &&
                board[row + 2][col + 2] == player &&
                board[row + 3][col + 3] == player) {
                return true;
            }
            if (row + 3 < ROWS && col - 3 >= 0 &&
                board[row][col] == player &&
                board[row + 1][col - 1] == player &&
                board[row + 2][col - 2] == player &&
                board[row + 3][col - 3] == player) {
                return true;
            }
        }
    }
    return false;
}

// Kernel to make a move for a given player (randomly select a column)
__global__ void make_move(int* board, int player, int* move_made, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool move_found = false;

    while (!move_found) {
        int col = curand(&states[idx]) % COLS;
        // Check if there's an empty spot in the selected column
        for (int row = ROWS - 1; row >= 0; --row) {
            if (board[row * COLS + col] == 0) {  // Empty spot found
                board[row * COLS + col] = player;
                *move_made = col;
                move_found = true;  // Valid move made
                break;
            }
        }
    }
}

// Initialize the random states
__global__ void init_random(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// Host function to print the board with Player 1 as 'X' and Player 2 as 'O'
void print_board(int board[ROWS][COLS]) {
    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS; ++col) {
            if (board[row][col] == 1) {
                std::cout << "X ";  // Player 1 move
            } else if (board[row][col] == 2) {
                std::cout << "O ";  // Player 2 move
            } else {
                std::cout << ". ";  // Empty space
            }
        }
        std::cout << std::endl;
    }
}

// Host function to check for win condition on the CPU
bool check_win_host(int board[ROWS][COLS], int player) {
    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS; ++col) {
            if (col + 3 < COLS && 
                board[row][col] == player && 
                board[row][col + 1] == player && 
                board[row][col + 2] == player && 
                board[row][col + 3] == player) {
                return true;
            }
            if (row + 3 < ROWS &&
                board[row][col] == player &&
                board[row + 1][col] == player &&
                board[row + 2][col] == player &&
                board[row + 3][col] == player) {
                return true;
            }
            if (row + 3 < ROWS && col + 3 < COLS &&
                board[row][col] == player &&
                board[row + 1][col + 1] == player &&
                board[row + 2][col + 2] == player &&
                board[row + 3][col + 3] == player) {
                return true;
            }
            if (row + 3 < ROWS && col - 3 >= 0 &&
                board[row][col] == player &&
                board[row + 1][col - 1] == player &&
                board[row + 2][col - 2] == player &&
                board[row + 3][col - 3] == player) {
                return true;
            }
        }
    }
    return false;
}

// Function to wait for key press
void wait_for_keypress() {
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.get();  // Wait for the user to press Enter
}

// Host function to play the game using two GPUs
void play_game() {
    int board[ROWS][COLS] = {0};  // Initialize empty board
    int* d_board[2];
    int move_made[2], *d_move_made[2];
    curandState* d_states[2];
    int current_player = 1;

    // Allocate memory for both GPUs
    for (int device = 0; device < 2; ++device) {
        cudaSetDevice(device);
        cudaMalloc(&d_board[device], ROWS * COLS * sizeof(int));
        cudaMalloc(&d_move_made[device], sizeof(int));
        cudaMalloc(&d_states[device], sizeof(curandState) * 1);
        cudaMemcpy(d_board[device], board, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);
        init_random<<<1, 1>>>(d_states[device], time(0) + device);  // Different seeds for each GPU
    }

    int turn = 1;
    while (true) {
        int device = current_player - 1;  // 0 for GPU 1, 1 for GPU 2

        // Set the correct device for the current player
        cudaSetDevice(device);

        // GPU makes a move
        make_move<<<1, 1>>>(d_board[device], current_player, d_move_made[device], d_states[device]);

        // Copy move result and board back to host
        cudaMemcpy(&move_made[device], d_move_made[device], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(board, d_board[device], ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);

        // Synchronize board state to the other GPU
        int other_device = (current_player == 1) ? 1 : 0;
        cudaSetDevice(other_device);
        cudaMemcpy(d_board[other_device], board, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);

        // Print the current board state
        std::cout << "Trun "<< turn <<": Player " << current_player << " (GPU " << device << ") made a move in column " << move_made[device] << std::endl;
        print_board(board);


        // Check for a win
        if (check_win_host(board, current_player)) {
            std::cout << "Player " << current_player << " (GPU " << device << ") wins!" << std::endl;
            break;
        }

        // Alternate players
        current_player = (current_player == 1) ? 2 : 1;

        // Wait for key press after the turn
        wait_for_keypress();

        turn++;
    }

    // Clean up
    for (int device = 0; device < 2; ++device) {
        cudaSetDevice(device);
        cudaFree(d_board[device]);
        cudaFree(d_move_made[device]);
        cudaFree(d_states[device]);
    }
}

int main() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    printf("Number of GPU Devices: %d\n", nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device Compute Major: %d Minor: %d\n", prop.major, prop.minor);
        printf("  Max Thread Dimensions: [%d][%d][%d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Device Clock Rate (KHz): %d\n", prop.clockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Registers Per Block: %d\n", prop.regsPerBlock);
        printf("  Registers Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
        printf("  Shared Memory Per Block: %zu\n", prop.sharedMemPerBlock);
        printf("  Shared Memory Per Multiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
        printf("  Total Constant Memory (bytes): %zu\n", prop.totalConstMem);
        printf("  Total Global Memory (bytes): %zu\n", prop.totalGlobalMem);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        // You can set the current chosen device property based on tracked min/max values
        
    }

    if (nDevices>=2) {
        play_game();
        return 0;
    }
    else {
        std::cout<< "You need 2 GPU devices to run the game."<<std::endl;
        return -1;
    }
}
