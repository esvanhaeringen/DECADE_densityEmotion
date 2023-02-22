#include "DECADE_GPU.h"
#include <iostream>
#include <fstream>

using namespace std;

void DECADE_GPU::writeStep()
{
	ofstream outputFile;
	outputFile.open(d_outputPath, std::ios_base::app);

	//one agent per row
	for (int idx = 0; idx != d_trackedIdcs.size(); ++idx)
	{
		outputFile << d_step << ',';
		outputFile << idx << ',';
		outputFile << c_valence[idx] << ',';
		outputFile << c_arousal[idx] << ',';
		outputFile << 0 << ',';
		outputFile << c_x[idx] << ',';
		outputFile << c_y[idx] << ',';
		outputFile << c_speed[idx] << ',';
		outputFile << c_densityM2[idx] << '\n';
	}
	outputFile.close();
}