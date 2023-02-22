#include "DECADE_GPU.h"
#include <device_launch_parameters.h>
#include "cuda_runtime.h"

DECADE_GPU::~DECADE_GPU() 
{
	//* GENERAL *
	cudaFree(c_attentionBias);
	cudaFree(c_attBiasStrength);
	cudaFree(c_beta);
	cudaFree(c_maxForce);
	cudaFree(c_densityImpact);
	cudaFree(c_pushImpact);

	//* AGENTS *
	//Body
	cudaFree(c_incomingForce);
	cudaFree(c_exertedForce);
	//Emotion
	cudaFree(c_valence);
	cudaFree(c_arousal);
	cudaFree(c_attPrefValence);
	cudaFree(c_attPrefArousal);
	cudaFree(c_regulationTime);
	cudaFree(c_maxRegulationTime);
	cudaFree(c_panicThreshold);
	//Contagion
	cudaFree(c_theta);
	cudaFree(c_channelStrength);
	cudaFree(c_deltaValence);
	cudaFree(c_deltaArousal);
}