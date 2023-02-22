#include "Durupinar.h"

using namespace std;

Durupinar::Durupinar(string const& configFile)
{
	setup(configFile);
}

//[ ACCESSORS ]
float Durupinar::panic(size_t const index) const
{
	return c_panic[index];
}
float Durupinar::combinedDose(size_t const self) const
{
	float combinedDose = 0;
	for (int idx = 0; idx < d_nDoses; ++idx)
		combinedDose += c_doses[self * d_nDoses + idx];
	return combinedDose;
}
bool Durupinar::infected(size_t const index) const
{
	return c_infected[index];
}