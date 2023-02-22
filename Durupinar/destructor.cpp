#include "Durupinar.h"
#include <device_launch_parameters.h>
#include "cuda_runtime.h"

Durupinar::~Durupinar()
{
	//* GENERAL *
	cudaFree(c_step);

	//* AGENTS *
	//Emotion
	cudaFree(c_panic);
	cudaFree(c_panicAppraisal);
	//Contagion
	cudaFree(c_doses);
	cudaFree(c_nDoses);
	cudaFree(c_meanDoseSize);
	cudaFree(c_infected);
	cudaFree(c_expressive);
	cudaFree(c_devStates);
}