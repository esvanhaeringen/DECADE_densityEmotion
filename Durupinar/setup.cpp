#include "Durupinar.h"
#include <fstream>
#include <random>

using namespace std;

void Durupinar::setupGeneral()
{
	BaseModel::setupGeneral();
	c_nDoses[0] = d_nDoses;
	c_meanDoseSize[0] = d_meanDoseSize;
}

void Durupinar::setupAgents()
{
	c_nAgents[0] = d_nAgents;
	for (size_t idx = 0; idx < d_nAgents; ++idx)
	{
		BaseModel::setupAgentBaseProp(idx);
		////Personality
		// epsilon = expressivity threshold of agent
		float mean = 0.5 - 0.5 * d_extravert[idx];
		c_epsilon[idx] = randomNormalLimited(mean, mean / 10, 0, 1);
		// delta = infection threshold of agent
		float const empathy = 0.354 * d_open[idx] + 0.177 * d_conscientious[idx] + 
			0.135 * d_extravert[idx] + 0.312 * d_agreeable[idx] + 0.021 * d_neurotic[idx];
		mean = 0.5 - 0.5 * empathy;
		c_delta[idx] = randomNormalLimited(mean, mean / 10, 0, 1);
		// lambda = regulation effeciency of agent
		c_lambda[idx] = d_neurotic[idx];
		//Emotion
		c_panic[idx] = d_panic[idx];
		c_panicAppraisal[idx] = 0;
		//Contagion
		for (int dose = 0; dose < d_nDoses; ++dose)
			c_doses[d_nDoses * idx + dose] = 0;
		c_infected[idx] = false;
		c_expressive[idx] = false;
	}
}

void Durupinar::setupOutput()
{
	ofstream outputFile;
	outputFile.open(d_outputPath);

	outputFile << "step" << ',';
	outputFile << "agent" << ',';
	outputFile << "valence" << ',';
	outputFile << "arousal" << ',';
	outputFile << "panic" << ',';
	outputFile << "x" << ',';
	outputFile << "y" << ',';
	outputFile << "speed" << ',';
	outputFile << "densityM2" << '\n';
	outputFile.close();
}