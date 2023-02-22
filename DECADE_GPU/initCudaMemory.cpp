#include "DECADE_GPU.h"
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

void DECADE_GPU::initCudaMemory()
{
	BaseModel::initCudaMemory();

	//| GENERAL |
	cudaMallocManaged(&c_attentionBias, sizeof(bool));
	cudaMallocManaged(&c_attBiasStrength, sizeof(float));
	cudaMallocManaged(&c_beta, sizeof(float));
	cudaMallocManaged(&c_maxForce, sizeof(float));
	cudaMallocManaged(&c_densityImpact, sizeof(float));
	cudaMallocManaged(&c_pushImpact, sizeof(float));

	//| AGENTS |
	//Body
	cudaMallocManaged(&c_incomingForce, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_exertedForce, d_nAgents * sizeof(float));
	//Emotion
	cudaMallocManaged(&c_valence, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_arousal, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_attPrefValence, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_attPrefArousal, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_regulationTime, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_maxRegulationTime, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_panicThreshold, d_nAgents * sizeof(float));
	//Contagion
	cudaMallocManaged(&c_theta, d_maxNeighbours * d_nAgents * sizeof(float));
	cudaMallocManaged(&c_channelStrength, d_maxNeighbours * d_nAgents * sizeof(float));
	cudaMallocManaged(&c_deltaValence, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_deltaArousal, d_nAgents * sizeof(float));
}