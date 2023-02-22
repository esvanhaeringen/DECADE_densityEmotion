#include "BaseModel.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace rapidxml;

void BaseModel::loadConfig(string const& configFile)
{
	ifstream file(configFile);
	rapidxml::xml_document<> doc;
	vector<char> buffer((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
	buffer.push_back('\0');
	doc.parse<0>(&buffer[0]);
	parseConfig(doc.first_node());
	doc.clear();
	file.close();
}

void BaseModel::parseConfig(xml_node<>* node)
{
	for (xml_node<>* type = node->first_node(); type; type = type->next_sibling())
	{
		if (strcmp(type->name(), "general") == 0)
			parseGeneral(type);
		else if (strcmp(type->name(), "agents") == 0)
			parseAgents(type);
		else if (strcmp(type->name(), "obstacles") == 0)
			parseObstacles(type);
	}
}

bool BaseModel::parseGeneralBaseProp(xml_node<>* property)
{
	if (strcmp(property->name(), "configName") == 0)
		d_configName = property->value();
	else if (strcmp(property->name(), "outputPath") == 0)
		d_outputPath = property->value();
	else if (strcmp(property->name(), "endStep") == 0)
		d_endStep = atoi(property->value());
	else if (strcmp(property->name(), "psOuterDist") == 0)
		d_psOuterDist = atof(property->value());
	else if (strcmp(property->name(), "psCritDist") == 0)
		d_psCritDist = atof(property->value());
	else if (strcmp(property->name(), "xSize") == 0)
		d_xSize = atof(property->value());
	else if (strcmp(property->name(), "ySize") == 0)
		d_ySize = atof(property->value());
	else if (strcmp(property->name(), "maxNeighbours") == 0)
		d_maxNeighbours = atoi(property->value());
	else if (strcmp(property->name(), "maxObstacles") == 0)
		d_maxObstacles = atoi(property->value());
	else if (strcmp(property->name(), "perceptDistance") == 0)
		d_perceptDistance = atof(property->value());
	else if (strcmp(property->name(), "fov") == 0)
		d_fov = atof(property->value());
	else if (strcmp(property->name(), "timeStep") == 0)
		d_timeStep = atof(property->value());
	else if (strcmp(property->name(), "walkSpeed") == 0)
		d_walkSpeed = atof(property->value());
	else if (strcmp(property->name(), "maxSpeed") == 0)
		d_maxSpeed = atof(property->value());
	else if (strcmp(property->name(), "eventStep") == 0)
		d_eventStep = atoi(property->value());
	else if (strcmp(property->name(), "eventX") == 0)
		d_eventX = atof(property->value());
	else if (strcmp(property->name(), "eventY") == 0)
		d_eventY = atof(property->value());
	else if (strcmp(property->name(), "eventIntensity") == 0)
		d_eventIntensity = atof(property->value());
	else if (strcmp(property->name(), "eventDistance") == 0)
		d_eventDistance = atof(property->value());
	else
		return false;
	return true;
}

void BaseModel::parseAgents(xml_node<>* node)
{
	for (xml_node<>* agent = node->first_node(); agent; agent = agent->next_sibling())
	{
		for (xml_node<>* property = agent->first_node(); property; property = property->next_sibling())
		{
			if (strcmp(property->name(), "personality") == 0)
				parsePersonality(property);
			else if (strcmp(property->name(), "emotion") == 0)
				parseEmotion(property);
			else if (strcmp(property->name(), "body") == 0)
				parseBody(property);
		}
		d_nAgents += 1;
	}
}

void BaseModel::parsePersonality(xml_node<>* node)
{
	for (xml_node<>* property = node->first_node(); property; property = property->next_sibling())
	{
		if (strcmp(property->name(), "open") == 0)
			d_open.push_back(atof(property->value()));
		else if (strcmp(property->name(), "conscientious") == 0)
			d_conscientious.push_back(atof(property->value()));
		else if (strcmp(property->name(), "extravert") == 0)
			d_extravert.push_back(atof(property->value()));
		else if (strcmp(property->name(), "agreeable") == 0)
			d_agreeable.push_back(atof(property->value()));
		else if (strcmp(property->name(), "neurotic") == 0)
			d_neurotic.push_back(atof(property->value()));
	}
}

void BaseModel::parseBody(xml_node<>* node)
{
	for (xml_node<>* property = node->first_node(); property; property = property->next_sibling())
	{
		if (strcmp(property->name(), "xPos") == 0)
			d_xPos.push_back(atof(property->value()));
		else if (strcmp(property->name(), "yPos") == 0)
			d_yPos.push_back(atof(property->value()));
		else if (strcmp(property->name(), "angle") == 0)
			d_angle.push_back(atof(property->value()));
		else if (strcmp(property->name(), "radius") == 0)
			d_radius.push_back(atof(property->value()));
		else if (strcmp(property->name(), "tracked") == 0)
		{
			d_tracked.push_back(atoi(property->value()));
			if (d_tracked[d_tracked.size() - 1] == true)
				d_trackedIdcs.push_back(d_tracked.size() - 1);
		}

	}
}

void BaseModel::parseObstacles(xml_node<>* node)
{
	for (xml_node<>* obstacle = node->first_node(); obstacle; obstacle = obstacle->next_sibling())
	{
		Obstacle newObstacle;
		for (xml_node<>* property = obstacle->first_node(); property; property = property->next_sibling())
		{
			if (strcmp(property->name(), "vertices") == 0)
				parseVertices(property, newObstacle);
			else if (strcmp(property->name(), "fillColour") == 0)
				newObstacle.fillCol = property->value();
			else if (strcmp(property->name(), "lineColour") == 0)
				newObstacle.lineCol = property->value();
			else if (strcmp(property->name(), "lineWidth") == 0)
				newObstacle.lineWidth = atof(property->value());
		}
		d_obstacles.push_back(newObstacle);
	}
}

void BaseModel::parseVertices(xml_node<>* node, Obstacle& newObstacle)
{
	for (xml_node<>* vertex = node->first_node(); vertex; vertex = vertex->next_sibling())
	{
		for (xml_node<>* property = vertex->first_node(); property; property = property->next_sibling())
		{
			if (strcmp(property->name(), "x") == 0)
				newObstacle.vrtcX.push_back(atof(property->value()));
			else if (strcmp(property->name(), "y") == 0)
				newObstacle.vrtcY.push_back(atof(property->value()));
			else if (strcmp(property->name(), "nextV") == 0)
				newObstacle.nextV.push_back(atoi(property->value()));
			else if (strcmp(property->name(), "prevV") == 0)
				newObstacle.prevV.push_back(atoi(property->value()));
		}
		newObstacle.nVrtcs += 1;
		d_nVrtcs += 1;
	}
}