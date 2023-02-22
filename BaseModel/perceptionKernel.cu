#include "BaseModel.h"
#include "mathFuncCuda.cuh"
#include "math_constants.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

//CUDA FUNCTION DEFINITIONS
//Kernels
__global__ void obstacleEdges(
	int* nAgents, int* nVrtcs, int* maxObstacles, float* x, float* y, float* vrtxPointX, 
	float* vrtxPointY, int* nextVrtcs, float* timeHorizonObst, float* prefSpeed, 
	float* radius, float* distObstacleEdges, int* nObstacleEdges, int* obstacleEdges);
__global__ void neighbours(
	int* nAgents, float* x, float* y, float* perceptDistance, int* maxNeighbours,
	int* nNeighbours, int* neighbours, float* distNeighbours, float* angle, float* fov);
__global__ void density(
	int* nAgents, int* maxNeighbours, int* nNeighbours, int* neighbours, 
	float* distNeighbours, float* radius, float* densityM2, float* psOuterDist);

void BaseModel::updatePerception()
{
	int blockSize = d_blocksGPU * 32;
	int numBlocks = (d_nAgents + blockSize - 1) / blockSize;

	obstacleEdges <<< numBlocks, blockSize >>> (
		c_nAgents, c_nVrtcs, c_maxObstacles, c_x, c_y, c_vrtxPointX, c_vrtxPointY, c_nextVrtcs,
		c_timeHorizonAgnt, c_prefSpeed, c_radius, c_distObstacleEdges, c_nObstacleEdges, c_obstacleEdges);
	cudaDeviceSynchronize();

	neighbours <<< numBlocks, blockSize >>> (
		c_nAgents, c_x, c_y, c_perceptDistance, c_maxNeighbours, c_nNeighbours, c_neighbours, 
		c_distNeighbours, c_angle, c_fov);
	cudaDeviceSynchronize();

	density <<< numBlocks, blockSize >>> (
		c_nAgents, c_maxNeighbours, c_nNeighbours, c_neighbours, c_distNeighbours, c_radius,
		c_densityM2, c_psOuterDist);
	cudaDeviceSynchronize();
}

__global__ void obstacleEdges(
	int* nAgents, int* nVrtcs, int* maxObstacles, float* x, float* y, float* vrtxPointX, float* vrtxPointY,
	int* nextVrtcs, float* timeHorizonObst, float* prefSpeed, float* radius, float* distObstacleEdges, 
	int* nObstacleEdges, int* obstacleEdges)
{
	int self = blockIdx.x * blockDim.x + threadIdx.x;
	int selfIdx = self * maxObstacles[0];

	if (self < nAgents[0]) //1 thread sorts distances of 1 agent
	{
		nObstacleEdges[self] = 0;

		for (int idx = 0; idx < nVrtcs[0]; ++idx)
		{
			int nextVertex = nextVrtcs[idx];
			float distSq = distSqPointLineSegment(vrtxPointX[idx], vrtxPointY[idx],
				vrtxPointX[nextVertex], vrtxPointY[nextVertex], x[self],
				y[self]);

			if (distSq < timeHorizonObst[0] * prefSpeed[self] * radius[self]) //if below maximum distance
			{
				if (nObstacleEdges[self] == 0)
				{
					distObstacleEdges[selfIdx + nObstacleEdges[self]] = distSq;
					obstacleEdges[selfIdx + nObstacleEdges[self]] = idx;
					++nObstacleEdges[self];
				}
				else if (distSq < distObstacleEdges[selfIdx + nObstacleEdges[self] - 1])
				{
					int prev = nObstacleEdges[self] - 2;

					if (nObstacleEdges[self] < maxObstacles[0])
					{
						distObstacleEdges[selfIdx + nObstacleEdges[self]] = distObstacleEdges[selfIdx + nObstacleEdges[self] - 1];
						obstacleEdges[selfIdx + nObstacleEdges[self]] = obstacleEdges[selfIdx + nObstacleEdges[self] - 1];
						++nObstacleEdges[self];
					}

					while (prev >= 0 && distSq < distObstacleEdges[selfIdx + prev])
					{
						distObstacleEdges[selfIdx + prev + 1] = distObstacleEdges[selfIdx + prev];
						obstacleEdges[selfIdx + prev + 1] = obstacleEdges[selfIdx + prev];
						--prev;
						if (prev < 0)   //to prevent going out of scope in check
							break;
					}

					distObstacleEdges[selfIdx + prev + 1] = distSq;
					obstacleEdges[selfIdx + prev + 1] = idx;

				}
				else if (nObstacleEdges[self] < maxObstacles[0])
				{
					distObstacleEdges[selfIdx + nObstacleEdges[self]] = distSq;
					obstacleEdges[selfIdx + nObstacleEdges[self]] = idx;
					++nObstacleEdges[self];
				}
			}
		}
	}
}

__global__ void neighbours(
	int* nAgents, float* x, float* y, float* perceptDistance, int* maxNeighbours, 
	int* nNeighbours, int* neighbours, float* distNeighbours, float* angle,
	float* fov)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self < nAgents[0]) //1 thread sorts neighbours of 1 agent
	{
		size_t const firstNeigh = self * maxNeighbours[0];
		nNeighbours[self] = 0;
		for (int idx = 0; idx < nAgents[0]; ++idx) 
		{
			if (idx == self)
				continue;
			float const euclDist = euclDistance(x[self], y[self], x[idx], y[idx]);
			if (euclDist < perceptDistance[0])
			{
				float const angleSelf2Other = atan2(y[idx] - y[self],
					x[idx] - x[self]) * 180 / CUDART_PI_F + 90;
				//if (angleInRange(angle[self], angleSelf2Other - fov[0] / 2, 
				//	angleSelf2Other + fov[0] / 2))
				if (true)
				{
					if (nNeighbours[self] == 0)
					{
						distNeighbours[firstNeigh] = euclDist;
						neighbours[firstNeigh] = idx;
						++nNeighbours[self];
					}
					else if (euclDist < distNeighbours[firstNeigh + nNeighbours[self] - 1])
					{
						int prev = nNeighbours[self] - 2;
						if (nNeighbours[self] < maxNeighbours[0])
						{
							distNeighbours[firstNeigh + nNeighbours[self]] =
								distNeighbours[firstNeigh + nNeighbours[self] - 1];
							neighbours[firstNeigh + nNeighbours[self]] = neighbours[firstNeigh +
								nNeighbours[self] - 1];
							++nNeighbours[self];
						}
						while (prev >= 0 && euclDist < distNeighbours[firstNeigh + prev])
						{
							distNeighbours[firstNeigh + prev + 1] = distNeighbours[firstNeigh + prev];
							neighbours[firstNeigh + prev + 1] = neighbours[firstNeigh + prev];
							--prev;
							if (prev < 0)   //to prevent going out of scope in check
								break;
						}
						distNeighbours[firstNeigh + prev + 1] = euclDist;
						neighbours[firstNeigh + prev + 1] = idx;
					}
					else if (nNeighbours[self] < maxNeighbours[0])
					{
						distNeighbours[firstNeigh + nNeighbours[self]] = euclDist;
						neighbours[firstNeigh + nNeighbours[self]] = idx;
						++nNeighbours[self];
					}
				}
			}
		}
	}
}

__global__ void density(
	int* nAgents, int* maxNeighbours, int* nNeighbours, int* neighbours, 
	float* distNeighbours, float* radius, float* densityM2, float* psOuterDist)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self < nAgents[0])
	{
		size_t const firstNeigh = self * maxNeighbours[0];
		float const radius1M2 = 2.91; //radius of a circle with a surface of 1 square metre
		densityM2[self] = 1; //include self in density
		for (size_t idx = 0; idx < nNeighbours[self]; ++idx)
		{
			size_t const other = neighbours[firstNeigh + idx];
			if ((distNeighbours[firstNeigh + idx] + radius[other]) <= radius1M2) //other is entirely within 1m2 circle around agent
				densityM2[self] += 1;
			else if ((distNeighbours[firstNeigh + idx] - radius[other]) <= radius1M2) //other is partially within 1m2 circle around agent
				densityM2[self] += (radius1M2 - (distNeighbours[firstNeigh + idx] - radius[other])) / (2 * radius[other]);
		}
	}
}