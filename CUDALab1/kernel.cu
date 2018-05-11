#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

// ---------------------------------
// BEGIN OF USER AREA

// Debug level, possible values: 0 - 5, 5 is highest
// Highest level will cause EXTREMELY detailed output (the whole array will be printed)
__constant__ const int DEBUG_LEVEL = 4;

// Array size for initialization, used only in inputArray functiont
__constant__ const int G_ARRAY_SIZE = 8192;

// Number of threads inside of block
__constant__ const int BLOCK_SIZE = 8;

int inputArray(int ** _arr) {
	int arr_size = G_ARRAY_SIZE;
	*_arr = new int[arr_size];

	for (int i = 0; i < arr_size; i++) {
		(*_arr)[i] = rand() % arr_size;
	}

	if (DEBUG_LEVEL >= 5) {
		std::wcout << "Array: ";
		for (int i = 0; i < arr_size; i++) {
			std::wcout << (*_arr)[i] << ", ";
		}
		std::wcout << std::endl;
	}


	return arr_size;
}

void outputArray(int * _arr, int arr_size) {
	if (DEBUG_LEVEL >= 5) {
		std::wcout << "Array: ";
		for (int i = 0; i < arr_size; i++) {
			std::wcout << _arr[i] << ", ";
		}
		std::wcout << std::endl;
	}
	
	bool sorted = true;
	for (int i = 1; i < arr_size; i++) {
		if (_arr[i] < _arr[i - 1]) {
			sorted = false;
			break;
		}
	}

	if (DEBUG_LEVEL >= 1) std::wcout << "Array sorting check, sorted: " << std::boolalpha << sorted << std::endl;
}

// END OF USER AREA
// ---------------------------------

// Number of blocks
__constant__ const int GRID_SIZE = G_ARRAY_SIZE / 2 / BLOCK_SIZE;

void pause() {
	std::wcout << "Press enter to continue . . . " << std::endl;
	std::cin.ignore();
}

bool inline cudaErrorOccured(cudaError_t _cudaStatus) {
	if (_cudaStatus != cudaSuccess) {
		std::wcout << std::endl << std::endl
			<< "------------------------------"
			<< "CUDA error: " << _cudaStatus << std::endl;
		if (DEBUG_LEVEL >= 1) std::wcout << cudaGetErrorString(_cudaStatus) << std::endl;
		std::wcout 
			<< "------------------------------"
			<< std::endl << std::endl;

		return true;
	}
	return false;
}

__device__ bool D_SORTED = false;

__device__ inline void swap(int * arr, int i, int j) {
	int tmp = arr[i];
	arr[i] = arr[j];
	arr[j] = tmp;
}

__global__ void kernel(int * arr, int parity) {
	//get own index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//array for swapping
	__shared__ int shared_arr[BLOCK_SIZE * 2];

	//copying forth	
	int last_deduction = 0;
	if (threadIdx.x == 0) {
		if (parity == 1 && blockIdx.x == GRID_SIZE - 1) last_deduction = 1;

		for (int i = 0; i < blockDim.x * 2 - last_deduction; i++) {
			shared_arr[i] = arr[2 * idx + i + parity];
		}
	}
	
	__syncthreads();

	// Last kernel shouldn't work in this case
	if (parity == 1 && idx == BLOCK_SIZE * GRID_SIZE - 1) return;

	//swapping
	if (shared_arr[threadIdx.x * 2] > shared_arr[threadIdx.x * 2 + 1]) {
		swap(shared_arr, threadIdx.x * 2, threadIdx.x * 2 + 1);
		D_SORTED = false;
	}

	__syncthreads();


	//copying back
	if (threadIdx.x == 0) {
		for (int i = 0; i < blockDim.x * 2 - last_deduction; i++) {
			arr[2 * idx + i + parity] = shared_arr[i];
		}
	}
}

void oddevensort(int * arr, int arr_size) {
	bool sorted = false;
	cudaError_t cudaStatus = cudaSuccess;
	int counter = 0;

	while (!sorted) {
		sorted = true;
		
		cudaStatus = cudaMemcpyToSymbol(D_SORTED, &sorted, sizeof(bool));
		if (cudaErrorOccured(cudaStatus)) return;

		kernel<<<GRID_SIZE, BLOCK_SIZE>>>(arr, 0);
		kernel<<<GRID_SIZE, BLOCK_SIZE>>>(arr, 1);

		cudaStatus = cudaMemcpyFromSymbol(&sorted, D_SORTED, sizeof(bool));
		if (cudaErrorOccured(cudaStatus)) return;
		counter++;
	}

	if (DEBUG_LEVEL >= 1) std::cout << "Sorting finished, iterations: " << counter << std::endl;
}

int main()
{
	cudaError_t cudaStatus = cudaSuccess;

	int arr_size = 0;
	int * arr = 0;
	int * d_arr = 0; //GPU copy of array

	//0. Выведение информации о CUDA device'ах
	if (DEBUG_LEVEL >= 1)
	{
		std::wcout << "CUDA realization of odd-even sorting algorithm" << std::endl;
		std::wcout << "Author: Roman Beltyukov" << std::endl << std::endl;

		std::wcout << "CUDA information" << std::endl;
		int deviceCount = 0;
		cudaStatus = cudaGetDeviceCount(&deviceCount);
		if (cudaErrorOccured(cudaStatus)) return 1;
		std::wcout << "Available CUDA device count: " << deviceCount << std::endl << std::endl;

		cudaDeviceProp devProps;
		for (int i = 0; i < deviceCount; i++) {
			cudaStatus = cudaGetDeviceProperties(&devProps, i);
			if (cudaErrorOccured(cudaStatus)) return 1;

			std::wcout
				<< "Device #" << i << ", CUDA version: " << devProps.major << "." << devProps.minor
				<< ", integrated: " << std::boolalpha << devProps.integrated << std::endl
				<< "Name: " << devProps.name << std::endl
				<< "Clockrate: " << (double)devProps.clockRate / 1024 << "MHz" << std::endl
				<< "Total global memory: " << (double)devProps.totalGlobalMem / 1024 / 1024 / 1024 << "GB" << std::endl
				<< "Shared memory per block: " << (double)devProps.sharedMemPerBlock / 1024 << "KB" << std::endl
				<< "Warp size: " << devProps.warpSize << std::endl
				<< "Max threads per block: " << devProps.maxThreadsPerBlock << std::endl
				<< "Max threads dimension: [" 
					<< devProps.maxThreadsDim[0] << ", " 
					<< devProps.maxThreadsDim[1] << ", " 
					<< devProps.maxThreadsDim[2] << "]" << std::endl
				<< "Max grid size: [" 
					<< devProps.maxGridSize[0] << ", " 
					<< devProps.maxGridSize[1] << ", " 
					<< devProps.maxGridSize[0] << "]" << std::endl
				<< std::endl;
		}
		std::wcout << std::endl;
	}

	//1. Получение массива
	arr_size = inputArray(&arr);
	if (DEBUG_LEVEL >= 1) std::wcout << "Array generated, size: " << arr_size << ", last element: " << arr[arr_size - 1] << std::endl;

	//2. Инициализация памяти на устройстве и копирование массива туда
	cudaStatus = cudaMalloc((void **)&D_SORTED, sizeof(bool));
	if (cudaErrorOccured(cudaStatus)) return 1;

	cudaStatus = cudaMalloc((void **)&d_arr, arr_size * sizeof(int));
	if (cudaErrorOccured(cudaStatus)) return 1;

	cudaStatus = cudaMemcpy(d_arr, arr, arr_size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaErrorOccured(cudaStatus)) return 1;

	if (DEBUG_LEVEL >= 1) std::wcout << "Memory allocation and copying host->device finished" << std::endl;

	//3. Сортировка
	oddevensort(d_arr, arr_size);

	cudaStatus = cudaGetLastError();
	if (cudaErrorOccured(cudaStatus)) return 1;

	cudaStatus = cudaDeviceSynchronize();
	if (cudaErrorOccured(cudaStatus)) return 1;


	//4. Копирование массива обратно
	cudaStatus = cudaMemcpy(arr, d_arr, arr_size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaErrorOccured(cudaStatus)) return 1;

	cudaStatus = cudaFree(d_arr);
	if (cudaErrorOccured(cudaStatus)) return 1;

	cudaStatus = cudaDeviceReset();;
	if (cudaErrorOccured(cudaStatus)) return 1;

	if (DEBUG_LEVEL >= 1) std::wcout << "Copying device->host and memory releasing finished" << std::endl;

	//5. Вывод массива
	outputArray(arr, arr_size);
	delete[] arr;
	if (DEBUG_LEVEL >= 1) std::wcout << "Array output finished" << std::endl;
	
	if (DEBUG_LEVEL >= 1) {
		std::wcout << "Program finished" << std::endl;
	}

	if (DEBUG_LEVEL >= 2) pause();
    return 0;
}

