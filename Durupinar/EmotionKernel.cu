#include "Durupinar.h"
#include "../BaseModel/mathFuncCuda.cuh"
#include "math_constants.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <random>

//CUDA FUNCTION DEFINITIONS
//Kernels
__global__ void decay(int* nAgents, float* panic, float* lambda, float* timeStep,
	bool* infected, float* delta);
__global__ void triggerEvent(int* nAgents, float* x, float* y, float* panicAppraisal,
	float* eventX, float* eventY, float* eventIntensity, float* eventDistance,
	bool* infected);
__global__ void contagion(int* nAgents, float* panic, float* doses,	int* nDoses, 
	float* meanDoseSize, bool* infected, bool* expressive, int* step, int* maxNeighbours,
	int* nNeighbours, int* neighbours, 	curandState* devStates, unsigned long seed);
__global__ void applyChange(int* nAgents, float* panic, float* panicAppraisal,
	float* doses, int* nDoses, float* delta, float* epsi, bool* infected, 
	bool* expressive, float* prefPersDist, float* critDist, float* prefSpeed, 
	float* walkSpeed, float* maxSpeed, float* timeStep);

void Durupinar::updateEmotion()
{
	int blockSize = d_blocksGPU * 32;
	int numBlocks = (d_nAgents + blockSize - 1) / blockSize;

	decay <<< numBlocks, blockSize >>> (
		c_nAgents, c_panic, c_lambda, c_timeStep, c_infected, c_delta);
	cudaDeviceSynchronize();

	if (d_step >= d_eventStep && d_step <= d_eventStep + 4)
	{
		triggerEvent <<< numBlocks, blockSize >>> (
			c_nAgents, c_x, c_y, c_panicAppraisal, c_eventX, c_eventY, c_eventIntensity,
			c_eventDistance, c_infected);
		cudaDeviceSynchronize();
	}
	c_step[0] = d_step;
	time_t seed;
	time(&seed);
	contagion <<< numBlocks, blockSize >>> (
		c_nAgents, c_panic, c_doses, c_nDoses, c_meanDoseSize, c_infected, c_expressive, 
		c_step, c_maxNeighbours, c_nNeighbours, c_neighbours, c_devStates, seed);
	cudaDeviceSynchronize();
	
	time(&seed);
	applyChange <<< numBlocks, blockSize >>> (
		c_nAgents, c_panic, c_panicAppraisal, c_doses, c_nDoses, c_delta, c_epsilon,
		c_infected, c_expressive, c_prefPersDist, c_psCritDist, c_prefSpeed, c_walkSpeed,
		c_maxSpeed, c_timeStep);
	cudaDeviceSynchronize();
}

//See equation 3 in Durupinar (2016) Psychological Parameters for Crowd Simulation.
//Note that the regulation effectiveness denoted as lambda here corresponds to beta
//in equation 3. This was chosen to prevent confusion with the parameter names of 
//DECADE.
__global__ void decay(int* nAgents, float* panic, float* lambda, float* timeStep,
	bool* infected, float* delta)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self > nAgents[0] - 1)
		return;
	panic[self] = fmaxf(panic[self] - panic[self] * lambda[self] * timeStep[0], 0);
	//infected[self] = panic[self] > delta[self];
}

__global__ void triggerEvent(int* nAgents, float* x, float* y, float* panicAppraisal,
	float* eventX, float* eventY, float* eventIntensity, float* eventDistance,
	bool* infected)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self > nAgents[0] - 1)
		return;
	float const dist = euclDistance(x[self], y[self], eventX[0], eventY[0]);
	panicAppraisal[self] += 1.f / pow((1.f + exp(dist - eventDistance[0])), 0.5);
	//if (panicAppraisal[self] > 0.5)
	//	infected[self] = true;
}

//See equation 4 in Durupinar (2016) Psychological Parameters for Crowd Simulation.
//Note that nDoses corresponds to the length of the memory indicated by k in the 
//equation.
__global__ void contagion(int* nAgents, float* panic, float* doses,	int* nDoses, 
	float* meanDoseSize, bool* infected, bool* expressive, int* step, int* maxNeighbours, 
	int* nNeighbours, int* neighbours, curandState* devStates, unsigned long seed)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self > nAgents[0] - 1)
		return;
	curand_init(seed, self, 0, &devStates[self]);
	
	float totalDose = 0;
	if (nNeighbours[self] > 0 && !infected[self])	//only contagion if there are neighbours and agent is susceptible
	{
		size_t const firstNeigh = self * maxNeighbours[0];
		for (size_t idx = 0; idx < nNeighbours[self]; ++idx)
		{
			size_t const other = neighbours[firstNeigh + idx];
			if (expressive[other] && infected[other])
				totalDose += fmaxf((curand_normal(&devStates[self]) *
					(meanDoseSize[0] / 10) + meanDoseSize[0]) * panic[other], 0);
		}
	}
	size_t const firstDose = nDoses[0] * self;
	doses[firstDose + (step[0] % nDoses[0])] = totalDose;
}

//See equation 1 and 5 in Durupinar (2016) Psychological Parameters for Crowd Simulation
__global__ void applyChange(int* nAgents, float* panic, float* panicAppraisal, 
	float* doses, int* nDoses, float* delta, float* epsi, bool* infected,
	bool* expressive, float* prefPersDist, float* critDist, float* prefSpeed, 
	float* walkSpeed, float* maxSpeed, float* timeStep)
{
	int self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self > nAgents[0] - 1)
		return;
	size_t const firstDose = nDoses[0] * self;
	float combinedDose = 0;
	for (size_t idx = 0; idx < nDoses[0]; ++idx)
		combinedDose += doses[firstDose + idx];

	//determine if susceptable agent becomes infected via contagion (equation 5), then apply equation 1
	if (combinedDose > delta[self] && !infected[self])
	{
		infected[self] = true;
		panic[self] = fminf(fmaxf(panic[self] + panicAppraisal[self] +
			combinedDose * timeStep[0], 0.f), 1.f);
	}
	else
		panic[self] = fminf(fmaxf(panic[self] + panicAppraisal[self], 0.f), 1.f);
	panicAppraisal[self] = 0; //reset for next step

	//to include the effects of appraisal and decay on infection status
	infected[self] = panic[self] > delta[self];

	//determine if agent is expressive based on extraversion of agent (epsi)
	expressive[self] = panic[self] > epsi[self];

	float const panicThreshold = delta[self];
	if (panic[self] > panicThreshold)
	{
		prefPersDist[self] = 0;
		prefSpeed[self] = ((panic[self] - panicThreshold) / (1 - panicThreshold)) * maxSpeed[0] * timeStep[0];
	}
	else
	{
		prefPersDist[self] = critDist[0];
		prefSpeed[self] = 0;
	}
}