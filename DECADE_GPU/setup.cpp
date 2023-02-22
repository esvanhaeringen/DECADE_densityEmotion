#include "DECADE_GPU.h"
#include <fstream>
#include <random>

using namespace std;

void DECADE_GPU::setupGeneral()
{
	BaseModel::setupGeneral();
	c_attentionBias[0] = d_attentionBias;
	c_attBiasStrength[0] = d_attBiasStrength;
	c_beta[0] = d_beta;
	c_maxForce[0] = d_maxForce;
	c_densityImpact[0] = d_densityImpact;
	c_pushImpact[0] = d_pushImpact;
}

void DECADE_GPU::setupAgents() 
{
	c_nAgents[0] = d_nAgents;
	for (size_t idx = 0; idx < d_nAgents; ++idx)
	{
		BaseModel::setupAgentBaseProp(idx);
		//Personality
		c_epsilon[idx] = 0.14 * d_open[idx] - 0.02 * d_conscientious[idx] + 0.32 * d_extravert[idx] + 0.11 * d_agreeable[idx] + 0.29 * d_neurotic[idx];;
		c_delta[idx] = 0.354 * d_open[idx] + 0.177 * d_conscientious[idx] + 0.135 * d_extravert[idx] + 0.312 * d_agreeable[idx] + 0.021 * d_neurotic[idx];
		c_lambda[idx] = 0.17 * d_open[idx] + 0.22 * d_conscientious[idx] + 0.19 * d_extravert[idx] + 0.45 * d_agreeable[idx] - 0.23 * d_neurotic[idx];
		//Emotion
		c_regulationTime[idx] = 0;
		c_maxRegulationTime[idx] = randomNormalLimited(d_maxRegulationTime - d_maxRegulationTime * c_lambda[idx],
			(d_maxRegulationTime - d_minRegulationTime) / 10,
		d_minRegulationTime, d_maxRegulationTime);
		c_panicThreshold[idx] = 0.5; //= squared distance to avoid sqrt
		//Emotion
		c_valence[idx] = d_valence[idx];
		c_arousal[idx] = d_arousal[idx];
		c_attPrefValence[idx] = d_attPrefValence[idx];
		c_attPrefArousal[idx] = d_attPrefArousal[idx];
		//Contagion
		c_deltaValence[idx] = 0;
		c_deltaArousal[idx] = 0;
	}
}

void DECADE_GPU::setupOutput()
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