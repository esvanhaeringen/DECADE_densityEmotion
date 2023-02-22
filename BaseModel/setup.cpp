#include "BaseModel.h"
#include <fstream>
#include <random>

using namespace std;

void BaseModel::setup(string const& configFile)
{
	loadConfig(configFile);
	initCudaMemory();
	setupGeneral();
	setupAgents();
	setupObstacles();
	if (d_outputPath != "")
		setupOutput();
	d_ready = true;
}

void BaseModel::setupGeneral()
{
	c_xSize[0] = d_xSize;
	c_ySize[0] = d_ySize;
	c_psOuterDist[0] = d_psOuterDist;
	c_psCritDist[0] = d_psCritDist;
	c_walkSpeed[0] = d_walkSpeed;
	c_maxSpeed[0] = d_maxSpeed;
	c_maxNeighbours[0] = d_maxNeighbours;
	c_maxObstacles[0] = d_maxObstacles;
	c_perceptDistance[0] = d_perceptDistance;
	c_fov[0] = d_fov;
	c_timeStep[0] = d_timeStep;
	c_timeHorizonObst[0] = 30.f;
	c_timeHorizonAgnt[0] = 30.f;
	c_eventStep[0] = d_eventStep;
	c_eventX[0] = d_eventX;
	c_eventY[0] = d_eventY;
	c_eventIntensity[0] = d_eventIntensity;
	c_eventDistance[0] = d_eventDistance;
}

void BaseModel::setupAgentBaseProp(size_t const idx)
{
	c_nAgents[0] = d_nAgents;
	//Body
	c_x[idx] = d_xPos[idx];
	c_y[idx] = d_yPos[idx];
	c_angle[idx] = d_angle[idx];
	c_veloX[idx] = 0;
	c_veloY[idx] = 0;
	c_speed[idx] = 0;
	c_radius[idx] = d_radius[idx];
	if (d_xPos[idx] < d_radius[idx] || d_xPos[idx] > d_xSize - d_radius[idx] ||
		d_yPos[idx] < d_radius[idx] || d_yPos[idx] > d_ySize - d_radius[idx])
		c_insideMap[idx] = false;
	else
		c_insideMap[idx] = true;
	//Perception
	c_nNeighbours[idx] = 0;
}

float BaseModel::randomNormalLimited(float mean, float stdDev, float lowerLimit, float upperLimit)
{
	std::random_device dev;
	std::mt19937 rng(dev());
	std::normal_distribution<> dist(mean, stdDev);
	float result = dist(rng);
	while (result < lowerLimit || result > upperLimit)
		result = dist(rng);
	return result;
}

void BaseModel::setupObstacles()
{
	c_nVrtcs[0] = d_nVrtcs;
	int count = 0;
	for (int idx = 0; idx < d_obstacles.size(); ++idx)
	{
		int startIdx = count;
		for (int v = 0; v < d_obstacles[idx].nVrtcs; ++v)
		{
			c_vrtxPointX[count] = d_obstacles[idx].vrtcX[v];
			c_vrtxPointY[count] = d_obstacles[idx].vrtcY[v];
			c_nextVrtcs[count] = startIdx + d_obstacles[idx].nextV[v];
			c_prevVrtcs[count] = startIdx + d_obstacles[idx].prevV[v];

			//we have to use vrtcX en vrtcY here instead of c_vrtxPointX en Y because next
			//vertex's coords are not yet inserted
			c_unitDirX[count] = norm2(
				d_obstacles[idx].vrtcX[d_obstacles[idx].nextV[v]] - d_obstacles[idx].vrtcX[v],
				d_obstacles[idx].vrtcY[d_obstacles[idx].nextV[v]] - d_obstacles[idx].vrtcY[v]);
			c_unitDirY[count] = norm2(
				d_obstacles[idx].vrtcY[d_obstacles[idx].nextV[v]] - d_obstacles[idx].vrtcY[v],
				d_obstacles[idx].vrtcX[d_obstacles[idx].nextV[v]] - d_obstacles[idx].vrtcX[v]);

			if (d_obstacles[idx].nVrtcs == 2)
				c_isConvex[count] = true;
			else
				c_isConvex[count] = leftOf2(
					d_obstacles[idx].vrtcX[d_obstacles[idx].prevV[v]], d_obstacles[idx].vrtcY[d_obstacles[idx].prevV[v]],
					d_obstacles[idx].vrtcX[v], d_obstacles[idx].vrtcY[v],
					d_obstacles[idx].vrtcX[d_obstacles[idx].nextV[v]], d_obstacles[idx].vrtcY[d_obstacles[idx].nextV[v]]) >= 0.0f;
			++count;
		}
	}
}

float BaseModel::leftOf2(float X1, float Y1, float X2, float Y2, float X3, float Y3)
{
	float compAX = X1 - X3;
	float compAY = Y1 - Y3;
	float compBX = X2 - X1;
	float compBY = Y2 - Y1;

	return compAX * compBY - compAY * compBX;
}

float BaseModel::norm2(float comp1, float comp2)
{
	return comp1 / sqrtf(comp1 * comp1 + comp2 * comp2);
}
