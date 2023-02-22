#include "ModelInterface.h"
#include "../rapidxml/rapidxml.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace rapidxml;

ModelInterface::ModelInterface(string configFile)
{
    ifstream file(configFile);
    rapidxml::xml_document<> doc;
    vector<char> buffer((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    buffer.push_back('\0');
    doc.parse<0>(&buffer[0]);
    string model = doc.first_node()->first_attribute("model")->value();
    doc.clear();
    file.close();

    if (model == "DECADE")
    {
        d_modelType = 1;
        d_decModel = new DECADE_GPU(configFile);
    }
    else if (model == "Durupinar")
    {
        d_modelType = 2;
        d_durModel = new Durupinar(configFile);
    }
    else
        throw;
}

void ModelInterface::update()
{
    if (d_modelType == 1)
        d_decModel->update();
    else if (d_modelType == 2)
        d_durModel->update();
}

int ModelInterface::modelType() const
{
    return d_modelType;
}
bool ModelInterface::ready() const
{
    if (d_modelType == 1)
        return d_decModel->ready();
    else if (d_modelType == 2)
        return d_durModel->ready();
}
int ModelInterface::currentStep() const
{
    if (d_modelType == 1)
        return d_decModel->currentStep();
    else if (d_modelType == 2)
        return d_durModel->currentStep();
}
int ModelInterface::endStep() const
{
    if (d_modelType == 1)
        return d_decModel->endStep();
    else if (d_modelType == 2)
        return d_durModel->endStep();
}
float ModelInterface::timeStep() const
{
    if (d_modelType == 1)
        return d_decModel->timeStep();
    else if (d_modelType == 2)
        return d_durModel->timeStep();
}
std::string const& ModelInterface::outputPath() const
{
    if (d_modelType == 1)
        return d_decModel->outputPath();
    else if (d_modelType == 2)
        return d_durModel->outputPath();
}
float ModelInterface::xSize() const
{
    if (d_modelType == 1)
        return d_decModel->xSize();
    else if (d_modelType == 2)
        return d_durModel->xSize();
}
float ModelInterface::ySize() const
{
    if (d_modelType == 1)
        return d_decModel->ySize();
    else if (d_modelType == 2)
        return d_durModel->ySize();
}
int ModelInterface::nObstacles() const
{
    if (d_modelType == 1)
        return d_decModel->nObstacles();
    else if (d_modelType == 2)
        return d_durModel->nObstacles();
}
int ModelInterface::nAgents() const
{
    if (d_modelType == 1)
        return d_decModel->nAgents();
    else if (d_modelType == 2)
        return d_durModel->nAgents();
}
float ModelInterface::x(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->x(index);
    else if (d_modelType == 2)
        return d_durModel->x(index);
}
float ModelInterface::y(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->y(index);
    else if (d_modelType == 2)
        return d_durModel->y(index);
}
float ModelInterface::speed(size_t index) const
{
    if (d_modelType == 1)
        return d_decModel->speed(index);
    else if (d_modelType == 2)
        return d_durModel->speed(index);
}
float ModelInterface::angle(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->angle(index);
    else if (d_modelType == 2)
        return d_durModel->angle(index);
}
float ModelInterface::radius(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->radius(index);
    else if (d_modelType == 2)
        return d_durModel->radius(index);
}
bool ModelInterface::insideMap(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->insideMap(index);
    else if (d_modelType == 2)
        return d_durModel->insideMap(index);
}
int ModelInterface::nNeighbours(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->nNeighbours(index);
    else if (d_modelType == 2)
        return d_durModel->nNeighbours(index);
}
int ModelInterface::neighbour(size_t const self, size_t const neighbour) const
{
    if (d_modelType == 1)
        return d_decModel->neighbour(self, neighbour);
    else if (d_modelType == 2)
        return d_durModel->neighbour(self, neighbour);
}
float ModelInterface::densityM2(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->densityM2(index);
    else if (d_modelType == 2)
        return d_durModel->densityM2(index);
}
float ModelInterface::incomingForce(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->incomingForce(index);
    else if (d_modelType == 2)
        return 0;
}
float ModelInterface::susceptibility(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->susceptibility(index);
    else if (d_modelType == 2)
        return d_durModel->susceptibility(index);
}
float ModelInterface::expressivity(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->expressivity(index);
    else if (d_modelType == 2)
        return d_durModel->expressivity(index);
}
float ModelInterface::regulationEfficiency(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->regulationEfficiency(index);
    else if (d_modelType == 2)
        return d_durModel->regulationEfficiency(index);
}
float ModelInterface::panic(size_t const index) const
{
    if (d_modelType == 1)
    {
        float dist = pow(-1 - d_decModel->valence(index), 2) +
            pow(1 - d_decModel->arousal(index), 2);
        return dist < 2 ? (2 - dist) / 2 : 0;
    }
    else if (d_modelType == 2)
        return d_durModel->panic(index);
}
float ModelInterface::combinedDose(size_t const self) const
{
    if (d_modelType == 1)
        return 0.f;
    else if (d_modelType == 2)
        return d_durModel->combinedDose(self);
}
bool ModelInterface::infected(size_t const index) const
{
    if (d_modelType == 1)
        return false;
    else if (d_modelType == 2)
        return d_durModel->infected(index);
}
Obstacle* ModelInterface::obstacle(size_t const index)
{
    if (d_modelType == 1)
        return d_decModel->obstacle(index);
    else if (d_modelType == 2)
        return d_durModel->obstacle(index);
}
float ModelInterface::test(size_t const index) const
{
    if (d_modelType == 1)
        return d_decModel->test(index);
    else if (d_modelType == 2)
        return d_durModel->test(index);
}