#include "Durupinar.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace rapidxml;

void Durupinar::parseGeneral(xml_node<>* node)
{
	for (xml_node<>* property = node->first_node(); property; property = property->next_sibling())
	{
		if (parseGeneralBaseProp(property))
			continue;
		else if (strcmp(property->name(), "nDoses") == 0)
			d_nDoses = atoi(property->value());
		else if (strcmp(property->name(), "meanDoseSize") == 0)
			d_meanDoseSize = atof(property->value());
	}
}

void Durupinar::parseEmotion(xml_node<>* node)
{
	for (xml_node<>* property = node->first_node(); property; property = property->next_sibling())
	{
		if (strcmp(property->name(), "panic") == 0)
			d_panic.push_back(atof(property->value()));
	}
}