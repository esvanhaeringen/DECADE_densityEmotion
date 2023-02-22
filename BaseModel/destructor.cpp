#include "BaseModel.h"
#include <device_launch_parameters.h>
#include "cuda_runtime.h"

BaseModel::~BaseModel()
{
	//* GENERAL *
	cudaFree(c_xSize);
	cudaFree(c_ySize);
	cudaFree(c_psOuterDist);
	cudaFree(c_psCritDist);
	cudaFree(c_walkSpeed);
	cudaFree(c_maxSpeed);
	cudaFree(c_maxNeighbours);
	cudaFree(c_maxObstacles);
	cudaFree(c_perceptDistance);
	cudaFree(c_fov);
	cudaFree(c_timeStep);
	cudaFree(c_timeHorizonObst);
	cudaFree(c_timeHorizonAgnt);
	cudaFree(c_eventStep);
	cudaFree(c_eventX);
	cudaFree(c_eventY);
	cudaFree(c_eventIntensity);
	cudaFree(c_eventDistance);
	//* AGENTS *
	cudaFree(c_nAgents);
	//Personality
	cudaFree(c_epsilon);
	cudaFree(c_delta);
	cudaFree(c_lambda);
	//Body
	cudaFree(c_x);
	cudaFree(c_y);
	cudaFree(c_veloX);
	cudaFree(c_veloY);
	cudaFree(c_speed);
	cudaFree(c_angle);
	cudaFree(c_radius);
	cudaFree(c_densityM2);
	cudaFree(c_insideMap);
	//Perception
	cudaFree(c_nNeighbours);
	cudaFree(c_neighbours);
	cudaFree(c_distNeighbours);
	cudaFree(c_nObstacleEdges);
	cudaFree(c_obstacleEdges);
	cudaFree(c_distObstacleEdges);
	//Navigation
	cudaFree(c_newVeloX);
	cudaFree(c_newVeloY);
	cudaFree(c_prefVeloX);
	cudaFree(c_prefVeloY);
	cudaFree(c_prefSpeed);
	cudaFree(c_nLines);
	cudaFree(c_nObstacleLines);
	cudaFree(c_orcaLines);
	cudaFree(c_projLines);
	cudaFree(c_prefPersDist);

	//* OBSTACLES *
	cudaFree(c_nVrtcs);
	cudaFree(c_vrtxPointX);
	cudaFree(c_vrtxPointY);
	cudaFree(c_unitDirX);
	cudaFree(c_unitDirY);
	cudaFree(c_prevVrtcs);
	cudaFree(c_nextVrtcs);
	cudaFree(c_isConvex);

	cudaFree(c_test);
}
