#pragma once
#include <string>
#include <vector>
#include "../rapidxml/rapidxml.hpp"
#include "../BaseModel/BaseModel.h"

typedef struct curandStateXORWOW curandState;

class Durupinar : public BaseModel
{
	//******************************| CPU VARIABLES |*********************************
	//* AGENTS *
	//Emotion
	std::vector<float> d_panic;
	//Contagion
	int d_nDoses;
	float d_meanDoseSize;

	//******************************| GPU VARIABLES |*********************************
	//* GENERAL *
	int* c_step;
	
	//* AGENTS *
	//Emotion
	float* c_panic;
	float* c_panicAppraisal;
	//Contagion
	float* c_doses;
	int* c_nDoses;
	float* c_meanDoseSize;
	bool* c_infected;
	bool* c_expressive;
	curandState* c_devStates;

public:
	Durupinar(std::string const& configFile);
	~Durupinar();

	//[ ACCESSORS ]
	float panic(size_t const index) const;
	float combinedDose(size_t const self) const;
	bool infected(size_t const index) const;

private:
	void parseGeneral(rapidxml::xml_node<>* node) override;
	void parseEmotion(rapidxml::xml_node<>* node) override;
	void initCudaMemory() override;
	void setupGeneral() override;
	void setupAgents() override;
	void updateEmotion() override;
	void setupOutput() override;
	void writeStep() override;
};