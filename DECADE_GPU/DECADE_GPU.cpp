#include "DECADE_GPU.h"

using namespace std;

DECADE_GPU::DECADE_GPU(string const& configFile)
{
	setup(configFile);
}

//[ ACCESSORS ]
float DECADE_GPU::valence(size_t const index) const
{
	return c_valence[index];
}
float DECADE_GPU::arousal(size_t const index) const
{
	return c_arousal[index];
}
float DECADE_GPU::beta() const
{
	return d_beta;
}
float DECADE_GPU::attPrefValence(size_t const index) const
{
	return c_attPrefValence[index];
}
float DECADE_GPU::attPrefArousal(size_t const index) const
{
	return c_attPrefArousal[index];
}
float DECADE_GPU::incomingForce(size_t const index) const
{
	return c_incomingForce[index];
}

//[ MODIFIERS ]

