#include "BaseModel.h"

using namespace std;

BaseModel::BaseModel()
{}

//[ ACCESSORS ]
bool BaseModel::ready() const
{
	return d_ready;
}
float BaseModel::timeStep() const
{
	return d_timeStep;
}
int BaseModel::currentStep() const
{
	return d_step;
}
int BaseModel::endStep() const
{
	return d_endStep;
}
std::string const& BaseModel::outputPath() const
{
	return d_outputPath;
}
float BaseModel::xSize() const
{
	return d_xSize;
}
float BaseModel::ySize() const
{
	return d_ySize;
}
int BaseModel::nAgents() const
{
	return d_nAgents;
}
int BaseModel::nObstacles() const
{
	return d_obstacles.size();
}
float BaseModel::x(size_t const index) const
{
	return c_x[index];
}
float BaseModel::y(size_t const index) const
{
	return c_y[index];
}
float BaseModel::speed(size_t const index) const
{
	return c_speed[index];
}
float BaseModel::angle(size_t const index) const
{
	return c_angle[index];
}
float BaseModel::radius(size_t const index) const
{
	return d_radius[index];
}
bool BaseModel::insideMap(size_t const index) const
{
	return c_insideMap[index];
}
int BaseModel::nNeighbours(size_t const index) const
{
	return c_nNeighbours[index];
}
int BaseModel::neighbour(size_t const self, size_t const neighbour) const
{
	return c_neighbours[self * d_maxNeighbours + neighbour];
}
float BaseModel::susceptibility(size_t const index) const
{
	return c_delta[index];
}
float BaseModel::expressivity(size_t const index) const
{
	return c_epsilon[index];
}
float BaseModel::regulationEfficiency(size_t const index) const
{
	return c_lambda[index];
}
float BaseModel::densityM2(size_t const index) const
{
	return c_densityM2[index];
}
Obstacle* BaseModel::obstacle(size_t const index)
{
	return &d_obstacles[index];
}


float BaseModel::test(size_t const index) const
{
	return c_test[index];
}
//[ MODIFIERS ]

