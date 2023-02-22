#include "Durupinar.h"
#include <iostream>
#include <fstream>

using namespace std;

void Durupinar::writeStep()
{
	ofstream outputFile;
	outputFile.open(d_outputPath, std::ios_base::app);

	//one agent per row
	for (int idx = 0; idx != d_trackedIdcs.size(); ++idx)
	{
		outputFile << d_step << ',';
		outputFile << idx << ',';
		outputFile << 0 << ',';
		outputFile << 0 << ',';
		outputFile << c_panic[idx] << ',';
		outputFile << c_x[idx] << ',';
		outputFile << c_y[idx] << ',';
		outputFile << c_speed[idx] << ',';
		outputFile << c_densityM2[idx] << '\n';
	}
	outputFile.close();
}