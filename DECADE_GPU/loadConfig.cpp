#include "DECADE_GPU.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace rapidxml;

void DECADE_GPU::parseGeneral(xml_node<>* node)
{
	for (xml_node<>* property = node->first_node(); property; property = property->next_sibling())
	{
		if (parseGeneralBaseProp(property))
			continue;
		else if (strcmp(property->name(), "contagion") == 0)
			d_contagion = atoi(property->value());
		else if (strcmp(property->name(), "attentionBias") == 0)
			d_attentionBias = atoi(property->value());
		else if (strcmp(property->name(), "attBiasStrength") == 0)
			d_attBiasStrength = atof(property->value());
		else if (strcmp(property->name(), "beta") == 0)
			d_beta = atof(property->value());
		else if (strcmp(property->name(), "maxForce") == 0)
			d_maxForce = atof(property->value());
		else if (strcmp(property->name(), "minRegulationTime") == 0)
			d_minRegulationTime = atof(property->value());
		else if (strcmp(property->name(), "maxRegulationTime") == 0)
			d_maxRegulationTime = atof(property->value());
		else if (strcmp(property->name(), "densityEffect") == 0)
			d_densityEffect = atoi(property->value());
		else if (strcmp(property->name(), "densityImpact") == 0)
			d_densityImpact = atof(property->value());
		else if (strcmp(property->name(), "pushImpact") == 0)
			d_pushImpact = atof(property->value());
	}
}

void DECADE_GPU::parseEmotion(xml_node<>* node)
{
	for (xml_node<>* property = node->first_node(); property; property = property->next_sibling())
	{
		if (strcmp(property->name(), "valence") == 0)
			d_valence.push_back(atof(property->value()));
		else if (strcmp(property->name(), "arousal") == 0)
			d_arousal.push_back(atof(property->value()));
		else if (strcmp(property->name(), "attPrefValence") == 0)
			d_attPrefValence.push_back(atof(property->value()));
		else if (strcmp(property->name(), "attPrefArousal") == 0)
			d_attPrefArousal.push_back(atof(property->value()));
	}
}