#include "Durupinar.h"
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

void Durupinar::initCudaMemory()
{
	BaseModel::initCudaMemory();

	//| GENERAL |
	cudaMallocManaged(&c_step, sizeof(int));
	cudaMallocManaged(&c_nDoses, sizeof(int));
	cudaMallocManaged(&c_meanDoseSize, sizeof(float));

	//| AGENTS |
	//Emotion
	cudaMallocManaged(&c_panic, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_panicAppraisal, d_nAgents * sizeof(float));
	//Contagion
	cudaMallocManaged(&c_doses, d_nAgents * d_nDoses * sizeof(float));
	cudaMallocManaged(&c_infected, d_nAgents * sizeof(bool));
	cudaMallocManaged(&c_expressive, d_nAgents * sizeof(bool));
	cudaMallocManaged(&c_devStates, d_nAgents * sizeof(curandState));
}