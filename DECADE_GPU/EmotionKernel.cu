#include "DECADE_GPU.h"
#include "../BaseModel/mathFuncCuda.cuh"
#include "math_constants.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <random>

//CUDA FUNCTION DEFINITIONS
//Kernels
__global__ void decay(
	int* nAgents, float* valence, float* arousal, float* regulationTime, 
	float* maxRegulationTime, float* timeStep, float* lambda);
__global__ void triggerEvent(
	int* nAgents, float* x, float* y, float* deltaValence, float* deltaArousal,
	float* eventX, float* eventY, float* eventIntensity, float* eventDistance);
__global__ void exertForce(int* nAgents, float* prefVeloX, float* prefVeloY, float* speed,
	float* maxForce, float* exertedForce, float* valence, float* arousal, 
	float* panicThreshold, float* test);
__global__ void collectIncomingForce(int* nAgents, float* timeStep, float* x, float* y,
	float* angle, float* radius, float* maxForce, float* exertedForce, float* incomingForce,
	float* valence, float* arousal, int* maxNeighbours, int* nNeighbours,
	int* neighbours, float* distNeighbours, float* psCritDist, int* maxObstacles,
	int* nObstacleEdges, int* obstacleEdges, float* vrtxPointX, float* vrtxPointY,
	int* nextVrtcs, float* pushImpact, float* test);
__global__ void densityEffect(
	int* nAgents, float* valence, float* arousal, float* deltaValence, float* deltaArousal,
	float* incomingForce, float* densityImpact, float* densityM2, int* maxNeighbours, 
	int* nNeighbours, int* neighbours, float* distNeighbours, float* radius, 
	float* psOuterDist, float* psCritDist);
__global__ void contagion(
	int* nAgents, float* valence, float* arousal, float* deltaValence, float* deltaArousal,
	int* maxNeighbours, int* nNeighbours, int* neighbours, float* distNeighbours,
	float* perceptDist,	float* channelStrength, float* epsi, float* delta, float* beta, 
	float* theta, bool* attentionBias, float* attBiasStrength, float* attPrefValence, 
	float* attPrefArousal);
__global__ void applyChange(int* nAgents, float* valence, float* arousal, float* deltaValence, 
	float* deltaArousal, float* prefPersDist, float* critDist, float* prefSpeed,
	float* walkSpeed, float* maxSpeed,	float* timeStep, float* panicThreshold);

void DECADE_GPU::updateEmotion()
{
	int blockSize = d_blocksGPU * 32;
	int numBlocks = (d_nAgents + blockSize - 1) / blockSize;

	decay <<< numBlocks, blockSize >>> (
		c_nAgents, c_valence, c_arousal, c_regulationTime, c_maxRegulationTime, c_timeStep,
		c_lambda);
	cudaDeviceSynchronize();

	if (d_step >= d_eventStep && d_step <= d_eventStep + 1)
	{
		triggerEvent <<< numBlocks, blockSize >>> (
			c_nAgents, c_x, c_y, c_deltaValence, c_deltaArousal, c_eventX, c_eventY, 
			c_eventIntensity, c_eventDistance);
		cudaDeviceSynchronize();
	}
	if (d_densityEffect)
	{
		exertForce <<< numBlocks, blockSize >>> (
			c_nAgents, c_prefVeloX, c_prefVeloY, c_speed, c_maxForce, c_exertedForce,
			c_valence, c_arousal, c_panicThreshold, c_test);
		cudaDeviceSynchronize();
		collectIncomingForce <<< numBlocks, blockSize >>> (
			c_nAgents, c_timeStep, c_x, c_y, c_angle, c_radius, c_maxForce, c_exertedForce, 
			c_incomingForce, c_valence, c_arousal, c_maxNeighbours, c_nNeighbours, 
			c_neighbours, c_distNeighbours, c_psCritDist, c_maxObstacles, c_nObstacleEdges,
			c_obstacleEdges, c_vrtxPointX, c_vrtxPointY, c_nextVrtcs, c_pushImpact, c_test);
		cudaDeviceSynchronize();
		densityEffect <<< numBlocks, blockSize >>> (
			c_nAgents, c_valence, c_arousal, c_deltaValence, c_deltaArousal, c_incomingForce,
			c_densityImpact, c_densityM2, c_maxNeighbours, c_nNeighbours, c_neighbours, 
			c_distNeighbours, c_radius, c_psOuterDist, c_psCritDist);
		cudaDeviceSynchronize();
	}
	if (d_contagion)
	{
		contagion <<< numBlocks, blockSize >>> (
			c_nAgents, c_valence, c_arousal, c_deltaValence, c_deltaArousal, c_maxNeighbours,
			c_nNeighbours, c_neighbours, c_distNeighbours, c_perceptDistance, c_channelStrength,
			c_epsilon, c_delta, c_beta, c_theta, c_attentionBias, c_attBiasStrength, c_attPrefValence, 
			c_attPrefArousal);
		cudaDeviceSynchronize();
	}
	applyChange <<< numBlocks, blockSize >>> (
		c_nAgents, c_valence, c_arousal, c_deltaValence, c_deltaArousal, c_prefPersDist, 
		c_psCritDist, c_prefSpeed, c_walkSpeed, c_maxSpeed, c_timeStep, c_panicThreshold);
	cudaDeviceSynchronize();


}

__global__ void decay(
	int* nAgents, float* valence, float* arousal, float* regulationTime, float* maxRegulationTime,
	float* timeStep, float* lambda)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self < nAgents[0])
	{
		float const regThreshold = 0.01; //=sqr(0.1) to avoid sqrt
		if (euclSqrDistance(valence[self], arousal[self], 0, 0) > regThreshold)
		{
			valence[self] -= valence[self] * (1.f + tanh(regulationTime[self] -
				maxRegulationTime[self])) / 2.f;
			arousal[self] -= arousal[self] * (1.f + tanh(regulationTime[self] -
				maxRegulationTime[self])) / 2.f;
			regulationTime[self] += timeStep[0];
		}
		else
			regulationTime[self] = 0;
	}
}

__global__ void triggerEvent(
	int* nAgents, float* x, float* y, float* deltaValence, float* deltaArousal,
	float* eventX, float* eventY, float* eventIntensity, float* eventDistance)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self > nAgents[0] - 1)
		return;
	float const dist = euclDistance(x[self], y[self], eventX[0], eventY[0]);
	deltaValence[self] -= 1.f / pow((1.f + exp(dist - eventDistance[0])), 0.5);
	deltaArousal[self] += 1.f / pow((1.f + exp(dist - eventDistance[0])), 0.5);
}


__global__ void exertForce(int* nAgents, float* prefVeloX, float* prefVeloY, float* speed,
	float* maxForce, float* exertedForce, float* valence, float* arousal, float* panicThreshold,
	float* test)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self >= nAgents[0])
		return;
	float const mass = 71.5; //see average for men and women in Hanson (2009)
	float const emotDist = euclSqrDistance(valence[self], arousal[self], -1, 1);
	test[self] = 0;
	exertedForce[self] = (emotDist < panicThreshold[0] && speed[self] < (0.5 * abs(prefVeloX[self], prefVeloY[self]))) ?
		((1 - emotDist / panicThreshold[0]) * maxForce[0]) / mass : 0;
	if (exertedForce[self] > 0)
		test[self] = 2;
}

__global__ void collectIncomingForce(int* nAgents, float* timeStep, float* x, float* y,
	float* angle, float* radius, float* maxForce, float* exertedForce, float* incomingForce,
	float* valence, float* arousal, int* maxNeighbours, int* nNeighbours,
	int* neighbours, float* distNeighbours, float* psCritDist, int* maxObstacles,
	int* nObstacleEdges, int* obstacleEdges, float* vrtxPointX, float* vrtxPointY,
	int* nextVrtcs, float* pushImpact, float* test)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self >= nAgents[0])
		return;               
	size_t const firstNeigh = self * maxNeighbours[0];
	incomingForce[self] = 0;
	float deltaX = 0;
	float deltaY = 0;
	float const meanArmLength = 0.816 * 5.15; //see average for men and women in Hanson (2009)
	for (size_t idx = 0; idx < nNeighbours[self]; ++idx)
	{
		size_t const other = neighbours[firstNeigh + idx];
		float const angleOther2Self = atan2(y[self] - y[other],
			x[self] - x[other]) * 180 / CUDART_PI_F + 90;
		if (distNeighbours[firstNeigh + idx] < (meanArmLength) && exertedForce[other] > 0 &&
			angleInRange(angle[other], angleOther2Self - 45, angleOther2Self + 45))
		{
			sin(angleOther2Self);
			valence[self] -= pushImpact[0] * incomingForce[self];
			arousal[self] += pushImpact[0] * incomingForce[self];
			incomingForce[self] += pushImpact[0] * incomingForce[self];
			deltaX += cos((angleOther2Self - 90) * CUDART_PI_F / 180) * exertedForce[other] * 5.15 * timeStep[0];
			deltaY += sin((angleOther2Self - 90) * CUDART_PI_F / 180) * exertedForce[other] * 5.15 * timeStep[0];
		}
	}
	if (deltaX == 0 && deltaY == 0) //no nett incoming force
		return;
	valence[self] = fminf(fmaxf(valence[self], -1.f), 1.f);
	arousal[self] = fminf(fmaxf(arousal[self], -1.f), 1.f);
	test[self] = 1;
	//deduce displacement distance if agent would otherwise be placed on top of a neighbour
	for (size_t idx = 0; idx < nNeighbours[self]; ++idx)
	{
		size_t const other = neighbours[firstNeigh + idx];
		float const newDist = euclDistance(x[self] + deltaX, y[self] + deltaY, x[other], y[other]);
		if (newDist < radius[self] + radius[other])
		{
			float const angleOther2Self = atan2(y[self] - y[neighbours[firstNeigh + other]],
				x[self] - x[neighbours[firstNeigh + other]]) * 180 / CUDART_PI_F - 90;
			deltaX -= cos((angleOther2Self + 90) * CUDART_PI_F / 180) * exertedForce[other] * 
				(radius[self] + radius[other] - newDist);
			deltaY -= sin((angleOther2Self + 90) * CUDART_PI_F / 180) * exertedForce[other] *
				(radius[self] + radius[other] - newDist);
		}
	}
	//for simplicity, prevent displacement entirely when this would result in crossing an obstacle
	for (int idx = 0; idx < nObstacleEdges[self]; ++idx)
	{
		size_t const v1 = obstacleEdges[self * maxObstacles[0] + idx];
		size_t const v2 = nextVrtcs[v1];
		if (intersect(x[self], y[self], x[self] + deltaX, y[self] + deltaY, 
			vrtxPointX[v1], vrtxPointY[v1], vrtxPointX[v2], vrtxPointY[v2]))
		{
			deltaX = 0;
			deltaY = 0;
		}
	}
	//apply displacement
	x[self] += deltaX;
	y[self] += deltaY;
}

__global__ void densityEffect(
	int* nAgents, float* valence, float* arousal, float* deltaValence, float* deltaArousal,
	float* incomingForce, float* densityImpact, float* densityM2, int* maxNeighbours, 
	int* nNeighbours, int* neighbours, float* distNeighbours, float* radius, 
	float* psOuterDist, float* psCritDist)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self > nAgents[0] - 1)
		return;
	size_t const firstNeigh = self * maxNeighbours[0];

	for (size_t idx = 0; idx < nNeighbours[self]; ++idx)
	{
		size_t const other = neighbours[firstNeigh + idx];
		float const dist = distNeighbours[firstNeigh + idx] - radius[other];
		if (dist < psCritDist[0])
		{
			deltaArousal[self] += densityImpact[0] * (1 - dist / psCritDist[0]);
			deltaValence[self] -= densityImpact[0] * (1 - dist / psCritDist[0]);
		}
	}
}

__global__ void contagion(
	int* nAgents, float* valence, float* arousal, float* deltaValence, float* deltaArousal,
	int* maxNeighbours, int* nNeighbours, int* neighbours, float* distNeighbours, float* perceptDist,
	float* channelStrength, float* epsi, float* delta, float* beta, float* theta,
	bool* attentionBias, float* attBiasStrength, float* attPrefValence, float* attPrefArousal)
{
	size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self > nAgents[0] - 1)
		return;

	if (nNeighbours[self] > 0)				//only contagion if there are neighbours
	{
		size_t const firstNeigh = self * maxNeighbours[0];

		//  [ CONNECTION STRENGTH ]
		//  First we determine the attention for each neighbour and the total 
		//  attention for all neighbours
		float totalTheta = 0.f;
		for (size_t idx = 0; idx < nNeighbours[self]; ++idx)
		{
			size_t const other = neighbours[firstNeigh + idx];
			if (attentionBias[0])
			{
				float const valenceComp = abs(valence[other]) * epsi[other] +
					attBiasStrength[0] * (1.f - attPrefValence[self]) * (1.f - (valence[other] + 1.f) / 2.f) +
					attBiasStrength[0] * attPrefValence[self] * (valence[other] + 1.f) / 2.f;
				float const arousalComp = abs(arousal[other]) * epsi[other] +
					attBiasStrength[0] * (1.f - attPrefArousal[self]) * (1.f - (arousal[other] + 1.f) / 2.f) +
					attBiasStrength[0] * attPrefArousal[self] * (arousal[other] + 1.f) / 2.f;
				theta[firstNeigh + idx] = euclDistance(valenceComp, arousalComp, 0.f, 0.f);
				totalTheta += theta[firstNeigh + idx];
			}
			else
			{
				theta[firstNeigh + idx] = epsi[other];
				totalTheta += theta[firstNeigh + idx];
			}
		}

		//  Then we calculate the connection strength (capital gamma in paper) 
		//  between the receiver and its neighbours.
		float totalConnect = 0.f;
		for (size_t idx = 0; idx < nNeighbours[self]; ++idx)
		{
			float const alpha = 1 - (distNeighbours[firstNeigh + idx] / perceptDist[0]);//1;//1 / distNeighbours[firstNeigh + idx];
			float const k1 = 1.f; //attention bias strength
			float const attBiasStrengh = 0.75;
			float const weighAtt = ((1.f - attBiasStrengh) / nAgents[0] + attBiasStrengh * theta[firstNeigh + idx]) /
				(1.f - attBiasStrengh + attBiasStrengh * k1 * totalTheta);
			channelStrength[firstNeigh + idx] = weighAtt * alpha * delta[self];
			totalConnect += channelStrength[firstNeigh + idx];
		}

		//  [ EMOTIONAL INFLUENCE ]
		//  Next we calculate the emotional influence per dimension. For first this the 
		//  emotion of the sender is weighed against the other sender.
		float weightedGroupValence = 0.f;
		float weightedGroupArousal = 0.f;
		for (size_t idx = 0; idx < nNeighbours[self]; ++idx)
		{
			int const other = neighbours[firstNeigh + idx];
			weightedGroupValence += (channelStrength[firstNeigh + idx] / totalConnect)
				* valence[other];
			weightedGroupArousal += (channelStrength[firstNeigh + idx] / totalConnect)
				* arousal[other];
		}
		//  Then the influence is calculated depending on which of three 
		//  conditions applies
		//[ VALENCE ]
		float influenceValence = 0.f;
		if (weightedGroupValence * valence[self] >= 0.f)
		{
			float const PI = 1.f - (1.f - abs(weightedGroupValence)) * (1.f -
				abs(valence[self]));
			float const NI = weightedGroupValence * valence[self];
			influenceValence = beta[0] * PI + (1.f - beta[0]) * NI;
			if (valence[self] < 0.f || weightedGroupValence < 0.f)
				influenceValence = -influenceValence;
			influenceValence -= valence[self];
		}
		else
			influenceValence = beta[0] * weightedGroupValence - (1.f -
				beta[self]) * valence[self];
		//[ AROUSAL ]
		float influenceArousal = 0.f;
		if (weightedGroupArousal * arousal[self] >= 0.f)
		{
			float const PI = 1.f - (1.f - abs(weightedGroupArousal)) * (1.f -
				abs(arousal[self]));
			float const NI = weightedGroupArousal * arousal[self];
			influenceArousal = beta[0] * PI + (1.f - beta[0]) * NI;
			if (arousal[self] < 0.f || weightedGroupArousal < 0.f)
				influenceArousal = -influenceArousal;
			influenceArousal -= arousal[self];
		}
		else
				influenceArousal = beta[0] * weightedGroupArousal - (1.f -
					beta[self]) * arousal[self];

		// [ EMOTIONAL CHANGE DUE TO CONTAGION ]
		// Lastly the change in emotion is determined. This is not applied directly 
		// to not affect the calculations of other agents that are performed in parallel.
		deltaValence[self] += totalConnect * influenceValence;
		deltaArousal[self] += totalConnect * influenceArousal;
	}
}

__global__ void applyChange(int* nAgents, float* valence, float* arousal, float* deltaValence,
	float* deltaArousal, float* prefPersDist, float* critDist, float* prefSpeed, 
	float* walkSpeed, float* maxSpeed, float* timeStep, float* panicThreshold)
{
	int self = blockIdx.x * blockDim.x + threadIdx.x;
	if (self > nAgents[0] - 1)
		return;

	valence[self] = fminf(fmaxf(valence[self] + deltaValence[self], -1.f), 1.f);
	arousal[self] = fminf(fmaxf(arousal[self] + deltaArousal[self], -1.f), 1.f);
	deltaValence[self] = 0; //reset for next step
	deltaArousal[self] = 0;

	float const emotDist = euclSqrDistance(valence[self], arousal[self], -1, 1);
	if (emotDist < panicThreshold[0])
	{
		prefPersDist[self] = 0;
		prefSpeed[self] = (1 - emotDist / panicThreshold[0]) * maxSpeed[0] * timeStep[0];
	}
	else
	{
		prefPersDist[self] = critDist[0];
		prefSpeed[self] = 0;
	}
}