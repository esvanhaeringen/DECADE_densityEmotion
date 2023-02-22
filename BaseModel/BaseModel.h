#pragma once
#include <string>
#include <vector>
#include "../rapidxml/rapidxml.hpp"

struct Obstacle;

class BaseModel
{
protected:
	//******************************| CPU VARIABLES |*********************************
	//* GENERAL *
	float d_xSize;						//world size in x dimension
	float d_ySize;						//world size in y dimension
	int d_step = 0;						//simulation step
	std::string d_configName;
	std::string d_outputPath;			//location + name + ext of output file
	int d_endStep = 0;					//step model stops running, 0 = forever
	int	d_nAgents = 0;  				//number of agents
	int	d_maxNeighbours;  				//maximum number of neighbour considered per agent
	int	d_maxObstacles;  				//maximum number of obstacles considered per agent
	float d_psOuterDist;
	float d_psCritDist;
	float d_walkSpeed;
	float d_maxSpeed;
	float d_perceptDistance;
	float d_fov;
	float d_timeStep;
	int d_eventStep;
	float d_eventX;
	float d_eventY;
	float d_eventIntensity;
	float d_eventDistance;
	bool d_ready = false;
	size_t d_blocksGPU = 8;

	//* AGENTS *
	//Personality
	std::vector<float> d_open;
	std::vector<float> d_conscientious;
	std::vector<float> d_extravert;
	std::vector<float> d_agreeable;
	std::vector<float> d_neurotic;
	//Body
	std::vector<float> d_xPos;
	std::vector<float> d_yPos;
	std::vector<float> d_angle;
	std::vector<float> d_radius;
	std::vector<bool> d_tracked;
	std::vector<int> d_trackedIdcs;

	//* OBSTACLES *
	std::vector<Obstacle> d_obstacles;
	int d_nVrtcs = 0;

	//******************************| GPU VARIABLES |*********************************
	//* GENERAL *
	float* c_xSize;
	float* c_ySize;
	float* c_psOuterDist;
	float* c_psCritDist;
	int* c_maxNeighbours;
	int* c_maxObstacles;
	float* c_perceptDistance;
	float* c_fov;
	float* c_timeStep;
	float* c_walkSpeed;
	float* c_maxSpeed;
	float* c_maxForce;
	float* c_timeHorizonObst;
	float* c_timeHorizonAgnt;
	int* c_eventStep;
	float* c_eventX;
	float* c_eventY;
	float* c_eventIntensity;
	float* c_eventDistance;

	//* AGENTS *
	int* c_nAgents;
	//Body
	float* c_x;
	float* c_y;
	float* c_veloX;
	float* c_veloY;
	float* c_speed;
	float* c_angle;
	float* c_radius;
	float* c_densityM2;
	bool* c_insideMap;
	//Perception
	int* c_nNeighbours;
	int* c_neighbours;
	float* c_distNeighbours;
	int* c_nObstacleEdges;
	int* c_obstacleEdges;
	float* c_distObstacleEdges;
	//Navigation
	float* c_newVeloX;
	float* c_newVeloY;
	float* c_prefVeloX;
	float* c_prefVeloY;
	float* c_prefSpeed;
	int* c_nLines;
	int* c_nObstacleLines;
	float* c_orcaLines;
	float* c_projLines;
	float* c_prefPersDist;
	//Personality
	float* c_epsilon;
	float* c_delta;
	float* c_lambda;

	//* OBSTACLES *
	int* c_nVrtcs;
	float* c_vrtxPointX;
	float* c_vrtxPointY;
	int* c_prevVrtcs;
	int* c_nextVrtcs;
	bool* c_isConvex;
	float* c_unitDirX;
	float* c_unitDirY;


	float* c_test;

public:
	BaseModel();
	~BaseModel();
	void setup(std::string const& configFile);
	void update();

	//[ ACCESSORS ]
	bool ready() const;
	float timeStep() const;
	int currentStep() const;
	int endStep() const;
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
	float susceptibility(size_t const index) const;
	float expressivity(size_t const index) const;
	float regulationEfficiency(size_t const index) const;
	Obstacle* obstacle(size_t const index);
	float test(size_t const index) const;

	//[ MODIFIERS ]

protected:
	void loadConfig(std::string const& configFile);
	void parseConfig(rapidxml::xml_node<>* node);
	virtual void parseGeneral(rapidxml::xml_node<>* node) {}
	bool parseGeneralBaseProp(rapidxml::xml_node<>* node);
	void parseAgents(rapidxml::xml_node<>* node);
	void parsePersonality(rapidxml::xml_node<>* node);
	virtual void parseEmotion(rapidxml::xml_node<>* node) {}
	void parseBody(rapidxml::xml_node<>* node);
	void parseObstacles(rapidxml::xml_node<>* node);
	void parseVertices(rapidxml::xml_node<>* node, Obstacle& newObstacle);
	virtual void initCudaMemory();
	virtual void setupGeneral();
	virtual void setupAgents() {}
	void setupAgentBaseProp(size_t const idx);
	float leftOf2(float X1, float Y1, float X2, float Y2, float X3, float Y3);
	float norm2(float comp1, float comp2);
	float randomNormalLimited(float mean, float stdDev, float lowerLimit, float upperLimit);
	void setupObstacles();
	void updatePerception();
	virtual void updateEmotion() {}
	void updateBehaviour();
	virtual void setupOutput() {}
	virtual void writeStep() {}
};

struct Obstacle
{
	std::vector<float> vrtcX;
	std::vector<float> vrtcY;
	std::vector<int> nextV;
	std::vector<int> prevV;
	int nVrtcs = 0;
	std::string fillCol;
	std::string lineCol;
	float lineWidth;
};