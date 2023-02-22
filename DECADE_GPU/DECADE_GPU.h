#pragma once
#include <string>
#include <vector>
#include "../rapidxml/rapidxml.hpp"
#include "../BaseModel/BaseModel.h"

class DECADE_GPU : public BaseModel
{
	//******************************| CPU VARIABLES |*********************************
	//* GENERAL *
	bool d_contagion;
	bool d_attentionBias;
	float d_attBiasStrength;
	float d_beta;
	float d_maxForce;
	float d_minRegulationTime;
	float d_maxRegulationTime;
	bool d_densityEffect;
	float d_densityImpact;
	float d_pushImpact;

	//* AGENTS *
	//Emotion
	std::vector<float> d_valence;
	std::vector<float> d_arousal;
	std::vector<float> d_attPrefValence;
	std::vector<float> d_attPrefArousal;

	//******************************| GPU VARIABLES |*********************************
	//* GENERAL *
	bool* c_attentionBias;
	float* c_attBiasStrength;
	float* c_beta;
	float* c_densityImpact;
	float* c_pushImpact;

	//* AGENTS *
	//Body
	float* c_incomingForce;
	float* c_exertedForce;
	//Emotion
	float* c_valence;
	float* c_arousal;
	float* c_attPrefValence;
	float* c_attPrefArousal;
	float* c_regulationTime;
	float* c_maxRegulationTime;
	float* c_panicThreshold;
	//Contagion
	float* c_theta;
	float* c_channelStrength;
	float* c_deltaValence;
	float* c_deltaArousal;

public:
	DECADE_GPU(std::string const& configFile);
	~DECADE_GPU();

	//[ ACCESSORS ]
	float valence(size_t const index) const;
	float arousal(size_t const index) const;
	float beta() const;
	float attPrefValence(size_t const index) const;
	float attPrefArousal(size_t const index) const;
	float incomingForce(size_t const index) const;

	//[ MODIFIERS ]

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

