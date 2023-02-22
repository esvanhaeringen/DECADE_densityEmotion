#pragma once
#include "GlWidget.h"
#include <QMainWindow>
#include <QWidget>
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include "../rapidxml/rapidxml.hpp"
#include "../rapidxml/rapidxml_print.hpp"

class GlWidget;
struct General
{
	std::string model = "DECADE";
	std::string configName;
	std::string outputPath;
	float xSize = 1000;
	float ySize = 1000;
	float timeStep = 0.04;
	int endStep = 1000;
	float perceptDistance = 5;
	float fov = 180;
	size_t maxNeighbours = 10;
	size_t maxObstacles = 10;
	bool densityEffect = true;
	float densityImpact = 0.1;
	float pushImpact = 0.1;
	float psOuterDist = 15;
	float psCritDist = 5;
	bool contagion = true;
	bool attentionBias = true;
	float attBiasStrength = 1;
	float beta = 0.5;
	int nDoses = 10;
	float meanDoseSize = 0.1;
	float maxRegulationTime = 100;
	float minRegulationTime = 10;
	float walkSpeed = 5;
	float maxSpeed = 15;
	float maxForce = 200;
	int eventStep = 50;
	float eventX = 500;
	float eventY = 500;
	float eventIntensity = 1;
	float eventDistance = 50;
};

struct Obstacle
{
	std::vector<float> vrtcX;
	std::vector<float> vrtcY;
	std::vector<int> nextV;
	std::vector<int> prevV;
	std::string fillCol = "#333333";
	std::string lineCol = "#000000";
	float lineWidth = 2;
};
struct Agent
{
	float open;
	float conscientious;
	float extravert;
	float agreeable;
	float neurotic;
	float x;
	float y;
	float angle;
	float radius;
	bool tracked;
	float valence;
	float arousal;
	float attPrefValence;
	float attPrefArousal;
	float panic;
};

namespace Ui {
	class ConfigBuilder;
}

class ConfigBuilder: public QWidget
{
	Q_OBJECT
	QString d_workDir;
	GlWidget* d_glWidget;
	General d_general;
	std::vector<Obstacle> d_obstacles;
	std::vector<Agent> d_agents;
	int d_selectedAgent = -1;
	int d_selectedObstacle = -1;
	int d_selectedVertex = -1;

public:
	ConfigBuilder(QWidget* parent = 0);
	~ConfigBuilder();
	void handleClick(float x, float y);

public slots:
	void p1_next();
	void p1_apply();
	void p1_loadConfig();
	
	void p2_previous();
	void p2_next();
	void p2_newObstacle();
	void p2_removePoint();
	void p2_removeObstacle();
	void p2_treeItemSelected();
	void p2_saveChanges();

	void p3_previous();
	void p3_build();
	void p3_generateAgents();
	void p3_saveChanges();
	void p3_removeSelectedAgent();
	void p3_removeAllAgents();
	void p3_listItemSelected();

	//[ ACCESSORS ]
	size_t page() const;
	General* general();
	Agent* agent(size_t idx);
	int nAgents() const;
	int selectedAgent() const;
	Obstacle* obstacle(size_t idx);
	int nObstacles() const;
	int selectedObstacle() const;
	int selectedVertex() const;

	//[ MODIFIERS ]
	void selectAgent(int idx);
	void selectObstacle(int idx);
	void selectVertex(int idx);

private:
	//Not page specific
	Ui::ConfigBuilder* ui;
	void update();

	//Page 1: General
	void loadGeneral();
	void saveGeneral();
	void parseConfig(rapidxml::xml_node<>* node);
	void parseGeneral(rapidxml::xml_node<>* node);
	void parseAgents(rapidxml::xml_node<>* node);
	void parsePersonality(rapidxml::xml_node<>* node, Agent& agent);
	void parseEmotion(rapidxml::xml_node<>* node, Agent& agent);
	void parseBody(rapidxml::xml_node<>* node, Agent& agent);
	void parseObstacles(rapidxml::xml_node<>* node);
	void parseVertices(rapidxml::xml_node<>* node, Obstacle& newObstacle);
	//Page 2: Obstacles
	void updateObstacleTree();
	void loadVrtxInfo();
	//Page 3: Agents
	bool placeAgent(Agent& agent, float x, float y);
	void assignPersonality(Agent& agent);
	void assignBody(Agent& agent);
	void assignEmotion(Agent& agent);
	void updateAgentList();
	void loadAgentInfo();
	void appendGeneral(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* root);
	void appendAgents(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* root);
	void appendPersonality(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* agent, size_t const idx);
	void appendEmotion(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* agent, size_t const idx);
	void appendBody(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* agent, size_t const idx);
	void appendObstacles(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* root);
	void appendVertices(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* obstacle, int idx);
	char* float2char(rapidxml::xml_document<>& doc, float value);
	char* int2char(rapidxml::xml_document<>& doc, int value);
	char* agentLabel(rapidxml::xml_document<>& doc, int value);
	char* obstacleLabel(rapidxml::xml_document<>& doc, int value);
	char* vertexLabel(rapidxml::xml_document<>& doc, int value);
};



