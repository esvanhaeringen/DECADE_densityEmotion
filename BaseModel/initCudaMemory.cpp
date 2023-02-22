#include "BaseModel.h"
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

void BaseModel::initCudaMemory()
{
	cudaError_t err0;

	//check that we have at least one CUDA device 
	int nDevices;
	err0 = cudaGetDeviceCount(&nDevices);
	if (err0 != cudaSuccess || nDevices == 0)
	{
		cerr << "ERROR: No CUDA device found\n";
		exit(1);
	}

	//select the first CUDA device as default
	err0 = cudaSetDevice(0);
	if (err0 != cudaSuccess)
	{
		cerr << "ERROR: Cannot set the chosen CUDA device\n";
		exit(1);
	}

	//| GENERAL |
	cudaMallocManaged(&c_xSize, sizeof(float));
	cudaMallocManaged(&c_ySize, sizeof(float));
	cudaMallocManaged(&c_psOuterDist, sizeof(float));
	cudaMallocManaged(&c_psCritDist, sizeof(float));
	cudaMallocManaged(&c_walkSpeed, sizeof(float));
	cudaMallocManaged(&c_maxSpeed, sizeof(float));
	cudaMallocManaged(&c_maxNeighbours, sizeof(int));
	cudaMallocManaged(&c_maxObstacles, sizeof(int));
	cudaMallocManaged(&c_perceptDistance, sizeof(float));
	cudaMallocManaged(&c_fov, sizeof(float));
	cudaMallocManaged(&c_timeStep, sizeof(float));
	cudaMallocManaged(&c_timeHorizonObst, sizeof(float));
	cudaMallocManaged(&c_timeHorizonAgnt, sizeof(float));
	cudaMallocManaged(&c_eventStep, sizeof(int));
	cudaMallocManaged(&c_eventX, sizeof(float));
	cudaMallocManaged(&c_eventY, sizeof(float));
	cudaMallocManaged(&c_eventIntensity, sizeof(float));
	cudaMallocManaged(&c_eventDistance, sizeof(float));

	//| AGENTS |
	cudaMallocManaged(&c_nAgents, sizeof(int));
	//Personality
	cudaMallocManaged(&c_epsilon, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_delta, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_lambda, d_nAgents * sizeof(float));
	//Body
	cudaMallocManaged(&c_x, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_y, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_veloX, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_veloY, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_speed, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_angle, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_radius, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_densityM2, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_insideMap, d_nAgents * sizeof(bool));
	//Perception
	cudaMallocManaged(&c_nNeighbours, d_nAgents * sizeof(int));
	cudaMallocManaged(&c_neighbours, d_maxNeighbours * d_nAgents * sizeof(int));
	cudaMallocManaged(&c_distNeighbours, d_maxNeighbours * d_nAgents * sizeof(int));
	cudaMallocManaged(&c_nObstacleEdges, d_nAgents * sizeof(int));
	cudaMallocManaged(&c_obstacleEdges, d_maxObstacles * d_nAgents * sizeof(int));
	cudaMallocManaged(&c_distObstacleEdges, d_maxObstacles * d_nAgents * sizeof(int));
	//Navigation
	cudaMallocManaged(&c_newVeloX, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_newVeloY, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_prefVeloX, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_prefVeloY, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_prefSpeed, d_nAgents * sizeof(float));
	cudaMallocManaged(&c_nLines, d_nAgents * sizeof(int));
	cudaMallocManaged(&c_nObstacleLines, d_nAgents * sizeof(int));
	cudaMallocManaged(&c_orcaLines, d_nAgents * 4 * (d_maxNeighbours + d_maxObstacles) * sizeof(float));
	cudaMallocManaged(&c_projLines, d_nAgents * 4 * (d_maxNeighbours + d_maxObstacles) * sizeof(float));
	cudaMallocManaged(&c_prefPersDist, d_nAgents * sizeof(float));

	//| OBSTACLES |
	cudaMallocManaged(&c_nVrtcs, sizeof(int));
	cudaMallocManaged(&c_vrtxPointX, d_nVrtcs * sizeof(float));
	cudaMallocManaged(&c_vrtxPointY, d_nVrtcs * sizeof(float));
	cudaMallocManaged(&c_prevVrtcs, d_nVrtcs * sizeof(int));
	cudaMallocManaged(&c_nextVrtcs, d_nVrtcs * sizeof(int));
	cudaMallocManaged(&c_isConvex, d_nVrtcs * sizeof(bool));
	cudaMallocManaged(&c_unitDirX, d_nVrtcs * sizeof(float));
	cudaMallocManaged(&c_unitDirY, d_nVrtcs * sizeof(float));


	cudaMallocManaged(&c_test, d_nAgents * sizeof(float));
}