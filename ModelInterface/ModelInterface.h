#include "../DECADE_GPU/DECADE_GPU.h"
#include "../Durupinar/Durupinar.h"

class ModelInterface
{
	DECADE_GPU* d_decModel = nullptr;
	Durupinar* d_durModel = nullptr;
	int d_modelType = 0;

public:
	ModelInterface(std::string const configFile);
	~ModelInterface();
	void update();

	//ACCESSORS
	int modelType() const;
	bool ready() const;
	int currentStep() const;
	int endStep() const;
	float timeStep() const;
	std::string const& outputPath() const;
	float xSize() const;
	float ySize() const;
	int nAgents() const;
	int nObstacles() const;
	float x(size_t const index) const;
	float y(size_t const index) const;
	float speed(size_t index) const;
	float angle(size_t const index) const;
	float radius(size_t const index) const;
	bool insideMap(size_t const index) const;
	int nNeighbours(size_t const index) const;
	int neighbour(size_t const self, size_t const neighbour) const;
	float densityM2(size_t const index) const;
	float incomingForce(size_t const index) const;
	float susceptibility(size_t const index) const;
	float expressivity(size_t const index) const;
	float regulationEfficiency(size_t const index) const;
	float panic(size_t const index) const;
	float combinedDose(size_t const self) const;
	bool infected(size_t const index) const;
	Obstacle* obstacle(size_t const index);
	float test(size_t const index) const;
};