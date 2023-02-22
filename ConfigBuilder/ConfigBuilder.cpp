#define _USE_MATH_DEFINES

#include "ConfigBuilder.h"
#include "ui_ConfigBuilder.h"
#include <qtreewidget.h>
#include <qlistwidget.h>
#include <QFileDialog>
#include <QMessageBox>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <exception>
#include "WinReg.hpp" 

using winreg::RegKey;
using winreg::RegException;
using winreg::RegExpected;
using namespace std;
using namespace rapidxml;


ConfigBuilder::ConfigBuilder(QWidget* parent)
    : QWidget(parent), ui(new Ui::ConfigBuilder)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);
    setWindowTitle(tr("Config builder: DECADE GPU"));
    setWindowIcon(QIcon(":/ConfigBuilder/icons/configBuilder.ico"));

    const wstring subKey = L"SOFTWARE\\DECADE GPU";
    RegKey key{ HKEY_CURRENT_USER, subKey };
    key.Open(HKEY_CURRENT_USER, subKey);
    if (auto expandSz2 = key.TryGetExpandStringValue(L"LastLocation"))
        d_workDir = QString::fromStdWString(expandSz2.GetValue());
    else
    {
        d_workDir = "";
        wcout << L"RegKey::TryGetLastLocation failed\n";
    }
    key.Close();

    d_glWidget = new GlWidget(this);
    d_glWidget->resetZoom();
    ui->GLholder->addWidget(d_glWidget);

    connect(ui->p1_next, SIGNAL(clicked()), this, SLOT(p1_next()));
    connect(ui->p1_apply, SIGNAL(clicked()), this, SLOT(p1_apply()));
    connect(ui->p1_loadConfig, SIGNAL(clicked()), this, SLOT(p1_loadConfig()));

    connect(ui->p2_previous, SIGNAL(clicked()), this, SLOT(p2_previous()));
    connect(ui->p2_next, SIGNAL(clicked()), this, SLOT(p2_next()));
    connect(ui->p2_newObstacle, SIGNAL(clicked()), this, SLOT(p2_newObstacle()));
    connect(ui->p2_removePoint, SIGNAL(clicked()), this, SLOT(p2_removePoint()));
    connect(ui->p2_removeObstacle, SIGNAL(clicked()), this, SLOT(p2_removeObstacle()));
    connect(ui->p2_treeWidget_obstacles, SIGNAL(itemSelectionChanged()), this, SLOT(p2_treeItemSelected()));
    connect(ui->p2_saveChanges, SIGNAL(clicked()), this, SLOT(p2_saveChanges()));

    connect(ui->p3_previous, SIGNAL(clicked()), this, SLOT(p3_previous()));
    connect(ui->p3_build , SIGNAL(clicked()), this, SLOT(p3_build()));
    connect(ui->p3_generateAgents, SIGNAL(clicked()), this, SLOT(p3_generateAgents()));
    connect(ui->p3_saveChanges, SIGNAL(clicked()), this, SLOT(p3_saveChanges()));
    connect(ui->p3_removeSelectedAgent, SIGNAL(clicked()), this, SLOT(p3_removeSelectedAgent()));
    connect(ui->p3_removeAllAgents, SIGNAL(clicked()), this, SLOT(p3_removeAllAgents()));
    connect(ui->p3_listWidgetAgents, SIGNAL(itemSelectionChanged()), this, SLOT(p3_listItemSelected()));

    loadGeneral();
}

ConfigBuilder::~ConfigBuilder()
{}

void ConfigBuilder::handleClick(float x, float y)
{
    if (ui->stackedWidget->currentIndex() == 1 && ui->p2_drawPoints->isChecked())
    {
        if (d_selectedObstacle == -1)
        {
            Obstacle tmp;
            d_obstacles.push_back(tmp);
            d_selectedObstacle = d_obstacles.size() - 1;
        }
        Obstacle& obstacle = d_obstacles[d_selectedObstacle];
        obstacle.vrtcX.push_back(x);
        obstacle.vrtcY.push_back(y);
        if (obstacle.nextV.size() != 0)
            obstacle.nextV[obstacle.nextV.size() - 1] = obstacle.nextV.size();
        obstacle.nextV.push_back(0);
        if (obstacle.prevV.size() != 0)
            obstacle.prevV[0] = obstacle.prevV.size();
        obstacle.prevV.push_back(obstacle.prevV.size() - 1);
        d_selectedVertex = obstacle.vrtcX.size() - 1;
        updateObstacleTree();
        loadVrtxInfo();
    }
    if (ui->stackedWidget->currentIndex() == 1 && !ui->p2_drawPoints->isChecked())
    {
        int selectedVertex = -1;
        int selectedObstacle = -1;
        float selectedDist = min(d_general.xSize, d_general.ySize) / 50;

        for (int obstacle = 0; obstacle < d_obstacles.size(); ++obstacle)
        {
            Obstacle& obst = d_obstacles[obstacle];
            for (int vertex = 0; vertex < obst.vrtcX.size(); ++vertex)
            {
                float dist = sqrt(pow(obst.vrtcX[vertex] - x, 2) +
                    pow(obst.vrtcY[vertex] - y, 2));
                if (dist < selectedDist)
                {
                    selectedObstacle = obstacle;
                    selectedVertex = vertex;
                    selectedDist = dist;
                }
            }
        }
        d_selectedVertex = selectedVertex;
        d_selectedObstacle = selectedObstacle;
        loadVrtxInfo();
        if (d_selectedVertex != -1)
            ui->p2_treeWidget_obstacles->setCurrentItem(ui->p2_treeWidget_obstacles->topLevelItem(d_selectedObstacle)->child(d_selectedVertex));
        else
        {
            ui->p2_treeWidget_obstacles->blockSignals(true);
            ui->p2_treeWidget_obstacles->clearSelection();
            ui->p2_treeWidget_obstacles->blockSignals(false);
        }
    }
    else if (ui->stackedWidget->currentIndex() == 2 && ui->p3_drawAgents->isChecked())
    {
        Agent newAgent;
        qDebug() << "click x " << x << " y " << y;
        if (placeAgent(newAgent, x, y))
        {
            qDebug() << "agent x " << newAgent.x << " y " << newAgent.y;
            assignPersonality(newAgent);
            assignBody(newAgent);
            assignEmotion(newAgent);
            d_agents.push_back(newAgent);
            d_selectedAgent = d_agents.size() - 1;
            updateAgentList();
            loadAgentInfo();
        }
        else
            qDebug() << "failed to place agent\n";
    }
    else if (ui->stackedWidget->currentIndex() == 2 && !ui->p3_drawAgents->isChecked())
    {
        int selectedAgent = -1;
        float selectedDist = min(d_general.xSize, d_general.ySize) / 50;

        for (int idx = 0; idx < d_agents.size(); ++idx)
        {
            float dist = sqrt(pow(d_agents[idx].x - x, 2) +
                pow(d_agents[idx].y - y, 2));
            if (dist < selectedDist)
            {
                selectedAgent = idx;
                selectedDist = dist;
            }
        }
        d_selectedAgent = selectedAgent;
        loadAgentInfo();
    }
}


void ConfigBuilder::saveGeneral()
{
    d_general.model = ui->comboBox_model->itemText(ui->comboBox_model->currentIndex()).toStdString();
    d_general.configName = ui->lineEdit_configName->text().toStdString();
    d_general.outputPath = ui->lineEdit_outputPath->text().toStdString();
    d_general.xSize = ui->doubleSpinBox_xSize->value();
    d_general.ySize = ui->doubleSpinBox_ySize->value();
    d_general.timeStep = ui->doubleSpinBox_timeStep->value();
    d_general.endStep = ui->spinBox_endStep->value();
    d_general.perceptDistance = ui->doubleSpinBox_percept->value();
    d_general.fov = ui->doubleSpinBox_fov->value();
    d_general.maxNeighbours = ui->spinBox_maxNeigh->value();
    d_general.maxObstacles = ui->spinBox_maxObst->value();
    d_general.densityEffect = ui->checkBox_density->isChecked();
    d_general.densityImpact = ui->doubleSpinBox_densityImpact->value();
    d_general.pushImpact = ui->doubleSpinBox_pushImpact->value();
    d_general.psOuterDist = ui->doubleSpinBox_psOuterDist->value();
    d_general.psCritDist = ui->doubleSpinBox_psCritDist->value();
    d_general.contagion = ui->checkBox_contagion->isChecked();
    d_general.attentionBias = ui->checkBox_bias->isChecked();
    d_general.attBiasStrength = ui->doubleSpinBox_attBiasStrength->value();
    d_general.beta = ui->doubleSpinBox_beta->value();
    d_general.nDoses = ui->spinBox_nDoses->value();
    d_general.meanDoseSize = ui->doubleSpinBox_meanDoseSize->value();
    d_general.minRegulationTime = ui->doubleSpinBox_minRegulationTime->value();
    d_general.maxRegulationTime = ui->doubleSpinBox_maxRegulationTime->value();
    d_general.walkSpeed = ui->doubleSpinBox_walkSpeed->value();
    d_general.maxSpeed = ui->doubleSpinBox_maxSpeed->value();
    d_general.maxForce = ui->doubleSpinBox_maxForce->value();
    d_general.eventStep = ui->spinBox_eventStep->value();
    d_general.eventX = ui->doubleSpinBox_eventX->value();
    d_general.eventY = ui->doubleSpinBox_eventY->value();
    d_general.eventIntensity = ui->doubleSpinBox_eventIntensity->value();
    d_general.eventDistance = ui->doubleSpinBox_eventDist->value();
    d_glWidget->setSize();
    d_glWidget->update();
}

void ConfigBuilder::loadGeneral()
{
    ui->comboBox_model->setCurrentIndex(ui->comboBox_model->findText(d_general.model.c_str()));
    ui->lineEdit_configName->setText(d_general.configName.c_str());
    ui->lineEdit_outputPath->setText(d_general.outputPath.c_str());
    ui->doubleSpinBox_xSize->setValue(d_general.xSize);
    ui->doubleSpinBox_ySize->setValue(d_general.ySize);
    ui->doubleSpinBox_timeStep->setValue(d_general.timeStep);
    ui->spinBox_endStep->setValue(d_general.endStep);
    ui->doubleSpinBox_percept->setValue(d_general.perceptDistance);
    ui->doubleSpinBox_fov->setValue(d_general.fov);
    ui->spinBox_maxNeigh->setValue(d_general.maxNeighbours);
    ui->spinBox_maxObst->setValue(d_general.maxObstacles);
    ui->doubleSpinBox_densityImpact->setValue(d_general.densityImpact);
    ui->doubleSpinBox_pushImpact->setValue(d_general.pushImpact);
    ui->doubleSpinBox_psOuterDist->setValue(d_general.psOuterDist);
    ui->doubleSpinBox_psCritDist->setValue(d_general.psCritDist);
    ui->doubleSpinBox_minRegulationTime->setValue(d_general.minRegulationTime);
    ui->doubleSpinBox_maxRegulationTime->setValue(d_general.maxRegulationTime);
    ui->doubleSpinBox_walkSpeed->setValue(d_general.walkSpeed);
    ui->doubleSpinBox_maxSpeed->setValue(d_general.maxSpeed);
    ui->spinBox_eventStep->setValue(d_general.eventStep);
    ui->doubleSpinBox_eventX->setValue(d_general.eventX);
    ui->doubleSpinBox_eventY->setValue(d_general.eventY);
    ui->doubleSpinBox_eventIntensity->setValue(d_general.eventIntensity);
    ui->doubleSpinBox_eventDist->setValue(d_general.eventDistance);
    ui->checkBox_density->setChecked(d_general.densityEffect);
    ui->checkBox_contagion->setChecked(d_general.contagion);
    ui->checkBox_bias->setChecked(d_general.attentionBias);
    ui->doubleSpinBox_attBiasStrength->setValue(d_general.attBiasStrength);
    ui->doubleSpinBox_beta->setValue(d_general.beta);
    ui->doubleSpinBox_maxForce->setValue(d_general.maxForce);
    ui->spinBox_nDoses->setValue(d_general.nDoses);
    ui->doubleSpinBox_meanDoseSize->setValue(d_general.meanDoseSize);
    if (d_general.model == "DECADE")
    {
        ui->checkBox_density->setEnabled(true);
        ui->lb1_density->setEnabled(true);
        ui->doubleSpinBox_densityImpact->setEnabled(true);
        ui->doubleSpinBox_pushImpact->setEnabled(true);
        ui->lb1_densityImpact->setEnabled(true);
        ui->lb1_pushImpact->setEnabled(true);
        ui->checkBox_contagion->setEnabled(true);
        ui->lb1_contagion->setEnabled(true);
        ui->checkBox_bias->setEnabled(true);
        ui->lb1_bias->setEnabled(true);
        ui->doubleSpinBox_attBiasStrength->setEnabled(true);
        ui->lb1_attBiasStrength->setEnabled(true);
        ui->doubleSpinBox_beta->setEnabled(true);
        ui->lb1_beta->setEnabled(true);
        ui->doubleSpinBox_maxForce->setEnabled(true);
        ui->lb1_maxForce->setEnabled(true);
        ui->doubleSpinBox_minRegulationTime->setEnabled(true);
        ui->lb1_minRegulationTime->setEnabled(true);
        ui->doubleSpinBox_maxRegulationTime->setEnabled(true);
        ui->lb1_maxRegulationTime->setEnabled(true);
        ui->spinBox_nDoses->setEnabled(false);
        ui->lb1_nDoses->setEnabled(false);
        ui->doubleSpinBox_meanDoseSize->setEnabled(false);
        ui->lb1_meanDoseSize->setEnabled(false);
    }
    else if (d_general.model == "Durupinar")
    {
        ui->checkBox_density->setEnabled(false);
        ui->lb1_density->setEnabled(false);
        ui->doubleSpinBox_densityImpact->setEnabled(false);
        ui->doubleSpinBox_pushImpact->setEnabled(false);
        ui->lb1_densityImpact->setEnabled(false);
        ui->lb1_pushImpact->setEnabled(false);
        ui->checkBox_contagion->setEnabled(false);
        ui->lb1_contagion->setEnabled(false);
        ui->checkBox_bias->setEnabled(false);
        ui->lb1_bias->setEnabled(false);
        ui->doubleSpinBox_attBiasStrength->setEnabled(false);
        ui->lb1_attBiasStrength->setEnabled(false);
        ui->doubleSpinBox_beta->setEnabled(false);
        ui->lb1_beta->setEnabled(false);
        ui->doubleSpinBox_maxForce->setEnabled(false);
        ui->lb1_maxForce->setEnabled(false);
        ui->doubleSpinBox_minRegulationTime->setEnabled(false);
        ui->lb1_minRegulationTime->setEnabled(false);
        ui->doubleSpinBox_maxRegulationTime->setEnabled(false);
        ui->lb1_maxRegulationTime->setEnabled(false);
        ui->spinBox_nDoses->setEnabled(true);
        ui->lb1_nDoses->setEnabled(true);
        ui->doubleSpinBox_meanDoseSize->setEnabled(true);
        ui->lb1_meanDoseSize->setEnabled(true);
    }
    d_glWidget->setSize();
    d_glWidget->update();
}

void ConfigBuilder::p1_loadConfig()
{
    QString filePath;
    filePath = QFileDialog::getOpenFileName(this, "Select config file", d_workDir,
        "Config Files (*.xml);;All files (*)");
    if (filePath != "")
    {
        d_agents.clear();
        d_obstacles.clear();
        ifstream file(filePath.toStdString());
        rapidxml::xml_document<> doc;
        vector<char> buffer((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
        buffer.push_back('\0');
        doc.parse<0>(&buffer[0]);
        d_general.model = doc.first_node()->first_attribute("model")->value();
        parseConfig(doc.first_node());
        doc.clear();
        file.close();
        loadGeneral();
        d_selectedAgent = -1;
        d_selectedObstacle = -1;
        d_selectedVertex = -1;

        const wstring subKey = L"SOFTWARE\\DECADE GPU";
        RegKey key{ HKEY_CURRENT_USER, subKey };
        key.Open(HKEY_CURRENT_USER, subKey);
        QFileInfo fi(filePath);
        d_workDir = fi.path();
        if (key.TrySetExpandStringValue(L"LastLocation", fi.path().toStdWString()).Failed())
            wcout << L"RegKey::TrySetLastLocation failed.\n";
    }
}

void ConfigBuilder::parseConfig(xml_node<>* node)
{
    for (xml_node<>* type = node->first_node(); type; type = type->next_sibling())
    {
        if (strcmp(type->name(), "general") == 0)
            parseGeneral(type);
        else if (strcmp(type->name(), "agents") == 0)
            parseAgents(type);
        else if (strcmp(type->name(), "obstacles") == 0)
            parseObstacles(type);
    }
}

void ConfigBuilder::parseGeneral(xml_node<>* node)
{
    for (xml_node<>* property = node->first_node(); property; property = property->next_sibling())
    {
        if (strcmp(property->name(), "configName") == 0)
            d_general.configName = property->value();
        else if (strcmp(property->name(), "outputPath") == 0)
            d_general.outputPath = property->value();
        else if (strcmp(property->name(), "endStep") == 0)
            d_general.endStep = atoi(property->value());
        else if (strcmp(property->name(), "densityEffect") == 0)
            d_general.densityEffect = atoi(property->value());
        else if (strcmp(property->name(), "densityImpact") == 0)
            d_general.densityImpact = atof(property->value());
        else if (strcmp(property->name(), "pushImpact") == 0)
            d_general.pushImpact = atof(property->value());
        else if (strcmp(property->name(), "psOuterDist") == 0)
            d_general.psOuterDist = atof(property->value());
        else if (strcmp(property->name(), "psCritDist") == 0)
            d_general.psCritDist = atof(property->value());
        else if (strcmp(property->name(), "contagion") == 0)
            d_general.contagion = atoi(property->value());
        else if (strcmp(property->name(), "attentionBias") == 0)
            d_general.attentionBias = atoi(property->value());
        else if (strcmp(property->name(), "attBiasStrength") == 0)
            d_general.attBiasStrength = atof(property->value());
        else if (strcmp(property->name(), "beta") == 0)
            d_general.beta = atof(property->value());
        else if (strcmp(property->name(), "nDoses") == 0)
            d_general.nDoses = atoi(property->value());
        else if (strcmp(property->name(), "meanDoseSize") == 0)
            d_general.meanDoseSize = atof(property->value());
        else if (strcmp(property->name(), "minRegulationTime") == 0)
            d_general.minRegulationTime = atof(property->value());
        else if (strcmp(property->name(), "maxRegulationTime") == 0)
            d_general.maxRegulationTime = atof(property->value());
        else if (strcmp(property->name(), "xSize") == 0)
            d_general.xSize = atof(property->value());
        else if (strcmp(property->name(), "ySize") == 0)
            d_general.ySize = atof(property->value());
        else if (strcmp(property->name(), "maxNeighbours") == 0)
            d_general.maxNeighbours = atoi(property->value());
        else if (strcmp(property->name(), "maxObstacles") == 0)
            d_general.maxObstacles = atoi(property->value());
        else if (strcmp(property->name(), "perceptDistance") == 0)
            d_general.perceptDistance = atof(property->value());
        else if (strcmp(property->name(), "fov") == 0)
            d_general.fov = atof(property->value());
        else if (strcmp(property->name(), "timeStep") == 0)
            d_general.timeStep = atof(property->value());
        else if (strcmp(property->name(), "walkSpeed") == 0)
            d_general.walkSpeed = atof(property->value());
        else if (strcmp(property->name(), "maxSpeed") == 0)
            d_general.maxSpeed = atof(property->value());
        else if (strcmp(property->name(), "maxForce") == 0)
            d_general.maxForce = atof(property->value());
        else if (strcmp(property->name(), "eventStep") == 0)
            d_general.eventStep = atoi(property->value());
        else if (strcmp(property->name(), "eventX") == 0)
            d_general.eventX = atof(property->value());
        else if (strcmp(property->name(), "eventY") == 0)
            d_general.eventY = atof(property->value());
        else if (strcmp(property->name(), "eventIntensity") == 0)
            d_general.eventIntensity = atof(property->value());
        else if (strcmp(property->name(), "eventDistance") == 0)
            d_general.eventDistance = atof(property->value());
    }
}

void ConfigBuilder::parseAgents(xml_node<>* node)
{
    for (xml_node<>* agent = node->first_node(); agent; agent = agent->next_sibling())
    {
        Agent newAgent;
        for (xml_node<>* property = agent->first_node(); property; property = property->next_sibling())
        {
            if (strcmp(property->name(), "personality") == 0)
                parsePersonality(property, newAgent);
            else if (strcmp(property->name(), "emotion") == 0)
                parseEmotion(property, newAgent);
            else if (strcmp(property->name(), "body") == 0)
                parseBody(property, newAgent);
        }
        d_agents.push_back(newAgent);
    }
}

void ConfigBuilder::parsePersonality(xml_node<>* node, Agent& agent)
{
    for (xml_node<>* property = node->first_node(); property; property = property->next_sibling())
    {
        if (strcmp(property->name(), "open") == 0)
            agent.open = atof(property->value());
        else if (strcmp(property->name(), "conscientious") == 0)
            agent.conscientious = atof(property->value());
        else if (strcmp(property->name(), "extravert") == 0)
            agent.extravert = atof(property->value());
        else if (strcmp(property->name(), "agreeable") == 0)
            agent.agreeable = atof(property->value());
        else if (strcmp(property->name(), "neurotic") == 0)
            agent.neurotic = atof(property->value());
    }
}

void ConfigBuilder::parseEmotion(xml_node<>* node, Agent& agent)
{
    for (xml_node<>* property = node->first_node(); property; property = property->next_sibling())
    {
        if (strcmp(property->name(), "valence") == 0)
            agent.valence = atof(property->value());
        else if (strcmp(property->name(), "arousal") == 0)
            agent.arousal = atof(property->value());
        else if (strcmp(property->name(), "attPrefValence") == 0)
            agent.attPrefValence = atof(property->value());
        else if (strcmp(property->name(), "attPrefArousal") == 0)
            agent.attPrefArousal = atof(property->value());
        else if (strcmp(property->name(), "panic") == 0)
            agent.panic = atof(property->value());
    }
}

void ConfigBuilder::parseBody(xml_node<>* node, Agent& agent)
{
    for (xml_node<>* property = node->first_node(); property; property = property->next_sibling())
    {
        if (strcmp(property->name(), "xPos") == 0)
            agent.x = atof(property->value());
        else if (strcmp(property->name(), "yPos") == 0)
            agent.y = atof(property->value());
        else if (strcmp(property->name(), "angle") == 0)
            agent.angle = atof(property->value());
        else if (strcmp(property->name(), "radius") == 0)
            agent.radius = atof(property->value());
        else if (strcmp(property->name(), "tracked") == 0)
            agent.tracked = atof(property->value());
    }
}

void ConfigBuilder::parseObstacles(xml_node<>* node)
{
    for (xml_node<>* obstacle = node->first_node(); obstacle; obstacle = obstacle->next_sibling())
    {
        Obstacle newObstacle;
        for (xml_node<>* property = obstacle->first_node(); property; property = property->next_sibling())
        {
            if (strcmp(property->name(), "vertices") == 0)
                parseVertices(property, newObstacle);
            else if (strcmp(property->name(), "fillColour") == 0)
                newObstacle.fillCol = property->value();
            else if (strcmp(property->name(), "lineCol") == 0)
                newObstacle.lineCol = property->value();
            else if (strcmp(property->name(), "lineWidth") == 0)
                newObstacle.lineWidth = atof(property->value());
        }
        d_obstacles.push_back(newObstacle);
    }
}

void ConfigBuilder::parseVertices(xml_node<>* node, Obstacle& newObstacle)
{
    for (xml_node<>* vertex = node->first_node(); vertex; vertex = vertex->next_sibling())
    {
        for (xml_node<>* property = vertex->first_node(); property; property = property->next_sibling())
        {
            if (strcmp(property->name(), "x") == 0)
                newObstacle.vrtcX.push_back(atof(property->value()));
            else if (strcmp(property->name(), "y") == 0)
                newObstacle.vrtcY.push_back(atof(property->value()));
            else if (strcmp(property->name(), "nextV") == 0)
                newObstacle.nextV.push_back(atoi(property->value()));
            else if (strcmp(property->name(), "prevV") == 0)
                newObstacle.prevV.push_back(atoi(property->value()));
        }
    }
}


void ConfigBuilder::updateObstacleTree()
{
    ui->p2_treeWidget_obstacles->blockSignals(true);
    ui->p2_treeWidget_obstacles->clear();
    ui->p2_treeWidget_obstacles->blockSignals(false);
    for (size_t obstacle = 0; obstacle < d_obstacles.size(); ++obstacle)
    {
        QTreeWidgetItem* obstacleItem = new QTreeWidgetItem(ui->p2_treeWidget_obstacles);
        obstacleItem->setText(0, QString("obstacle ") + QString::number(obstacle));
        for (size_t vrtx = 0; vrtx < d_obstacles[obstacle].vrtcX.size(); ++vrtx)
        {
            QTreeWidgetItem* vrtxItem = new QTreeWidgetItem(obstacleItem);
            vrtxItem->setText(0, QString("vrtx ") + QString::number(vrtx));
        }
        ui->p2_treeWidget_obstacles->insertTopLevelItem(obstacle, obstacleItem);
    }
    
    if (d_selectedObstacle != -1)
    {
        if (d_selectedVertex == -1)
        {
            ui->p2_treeWidget_obstacles->setCurrentItem(ui->p2_treeWidget_obstacles->topLevelItem(d_selectedObstacle));
            ui->p2_treeWidget_obstacles->expandItem(ui->p2_treeWidget_obstacles->currentItem());
        }
        else
        {
            ui->p2_treeWidget_obstacles->setCurrentItem(ui->p2_treeWidget_obstacles->topLevelItem(d_selectedObstacle)->child(d_selectedVertex));
            ui->p2_treeWidget_obstacles->expandItem(ui->p2_treeWidget_obstacles->currentItem()->parent());
        }
    }
    else
        ui->p2_treeWidget_obstacles->clearSelection();
}

void ConfigBuilder::loadVrtxInfo()
{
    if (d_selectedObstacle == -1)
    {
        ui->label_selectedObstacle->setEnabled(false);
        ui->lineEdit_fillCol->setEnabled(false);
        ui->lineEdit_lineCol->setEnabled(false);
        ui->doubleSpinBox_lineWidth->setEnabled(false);
        ui->p2_saveChanges->setEnabled(false);
        ui->label_selectedObstacle->setText("-1");
        ui->lineEdit_fillCol->setText("");
        ui->lineEdit_lineCol->setText("");
        ui->doubleSpinBox_lineWidth->setValue(0);
    }
    else
    {
        ui->label_selectedObstacle->setEnabled(true);
        ui->lineEdit_fillCol->setEnabled(true);
        ui->lineEdit_lineCol->setEnabled(true);
        ui->doubleSpinBox_lineWidth->setEnabled(true);
        ui->p2_saveChanges->setEnabled(true);
        ui->label_selectedObstacle->setText(QString::number(d_selectedObstacle));
        ui->lineEdit_fillCol->setText(QString::fromStdString(d_obstacles[d_selectedObstacle].fillCol));
        ui->lineEdit_lineCol->setText(QString::fromStdString(d_obstacles[d_selectedObstacle].lineCol));
        ui->doubleSpinBox_lineWidth->setValue(d_obstacles[d_selectedObstacle].lineWidth);
    }
    if (d_selectedVertex == -1)
    {
        ui->label_selectedVrtx->setEnabled(false);
        ui->doubleSpinBox_vrtxX->setEnabled(false);
        ui->doubleSpinBox_vrtxY->setEnabled(false);
        ui->label_vrtxNext->setEnabled(false);
        ui->label_vrtxPrevious->setEnabled(false);
        ui->label_selectedVrtx->setText("-1");
        ui->doubleSpinBox_vrtxX->setValue(0);
        ui->doubleSpinBox_vrtxY->setValue(0);
        ui->label_vrtxNext->setText("0");
        ui->label_vrtxPrevious->setText("0");
    }
    else
    {
        ui->label_selectedVrtx->setEnabled(true);
        ui->doubleSpinBox_vrtxX->setEnabled(true);
        ui->doubleSpinBox_vrtxY->setEnabled(true);
        ui->label_vrtxNext->setEnabled(true);
        ui->label_vrtxPrevious->setEnabled(true);
        ui->lineEdit_fillCol->setEnabled(true);
        ui->lineEdit_lineCol->setEnabled(true);
        ui->doubleSpinBox_lineWidth->setEnabled(true);
        ui->label_selectedVrtx->setText(QString::number(d_selectedVertex));
        ui->doubleSpinBox_vrtxX->setValue(d_obstacles[d_selectedObstacle].vrtcX[d_selectedVertex]);
        ui->doubleSpinBox_vrtxY->setValue(d_obstacles[d_selectedObstacle].vrtcY[d_selectedVertex]);
        ui->label_vrtxNext->setText(QString::number(d_obstacles[d_selectedObstacle].nextV[d_selectedVertex]));
        ui->label_vrtxPrevious->setText(QString::number(d_obstacles[d_selectedObstacle].prevV[d_selectedVertex]));
    }
    d_glWidget->update();
}


bool ConfigBuilder::placeAgent(Agent& agent, float x, float y)
{
    for (int obstacle = 0; obstacle < d_obstacles.size(); ++obstacle)
    {
        int count = 0;
        Obstacle& obst = d_obstacles[obstacle];
        for (int vertex = 0; vertex < d_obstacles[obstacle].vrtcX.size(); ++vertex)
        {
            float a = (obst.vrtcY[vertex] - obst.vrtcY[obst.nextV[vertex]]) / (obst.vrtcX[vertex] - obst.vrtcX[obst.nextV[vertex]]);
            float b = obst.vrtcY[vertex] - (a * obst.vrtcX[vertex]);
            if (abs(a) == std::numeric_limits<float>::infinity())
            {
                a = 1;
                b = y - obst.vrtcX[vertex];
            }
            //check if agent is too close to an obstacle line in the x direction
            if (((obst.vrtcY[vertex] > y && obst.vrtcY[obst.nextV[vertex]] < y) ||
                (obst.vrtcY[vertex] < y && obst.vrtcY[obst.nextV[vertex]] > y)) && 
                (abs(x - ((y - b) / a)) < ui->doubleSpinBox_genRadius->value())) //to get x on line ax+b from a given y: x=(y-b)/a
                return false;
            //check if agent is too close to an obstacle line in the y direction
            if (((obst.vrtcX[vertex] > x && obst.vrtcX[obst.nextV[vertex]] < x) ||
                (obst.vrtcX[vertex] < x && obst.vrtcX[obst.nextV[vertex]] > x)) &&
                (abs(y - (a * x + b)) < ui->doubleSpinBox_genRadius->value()))
                return false;
            //count number of lines that are crossed to the right of the agent if x is maintained the same
            if (((obst.vrtcY[vertex] > y && obst.vrtcY[obst.nextV[vertex]] < y) ||
                (obst.vrtcY[vertex] < y && obst.vrtcY[obst.nextV[vertex]] > y)) &&
                (x < ((y - b) / a))) 
                count += 1;
        }
        if (count % 2 == 1) //uneven number means the agent is inside the obstacle
            return false;
    }
    for (int other = 0; other < d_agents.size(); ++other)
        if (sqrt(pow(x - d_agents[other].x, 2) + pow(y - d_agents[other].y, 2)) < (ui->doubleSpinBox_genRadius->value() + 
            d_agents[other].radius + ui->doubleSpinBox_persSpace->value()))
            return false;
    agent.x = x;
    agent.y = y;
    return true;
}

void ConfigBuilder::assignPersonality(Agent& agent)
{
    if (ui->radioButton_persManual->isChecked())
    {
        agent.open = ui->doubleSpinBox_genO->value();
        agent.conscientious = ui->doubleSpinBox_genC->value();
        agent.extravert = ui->doubleSpinBox_genE->value();
        agent.agreeable = ui->doubleSpinBox_genA->value();
        agent.neurotic = ui->doubleSpinBox_genN->value();
    }
    else if (ui->radioButton_persNorm->isChecked())
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::normal_distribution<> distO(ui->doubleSpinBox_genO->value(), ui->doubleSpinBox_sdO->value());
        std::normal_distribution<> distC(ui->doubleSpinBox_genC->value(), ui->doubleSpinBox_sdC->value());
        std::normal_distribution<> distE(ui->doubleSpinBox_genE->value(), ui->doubleSpinBox_sdE->value());
        std::normal_distribution<> distA(ui->doubleSpinBox_genA->value(), ui->doubleSpinBox_sdA->value());
        std::normal_distribution<> distN(ui->doubleSpinBox_genN->value(), ui->doubleSpinBox_sdN->value());
        agent.open = distO(rng);
        agent.conscientious = distC(rng);
        agent.extravert = distE(rng);
        agent.agreeable = distA(rng);
        agent.neurotic = distN(rng);
    }
    else if (ui->radioButton_persUni->isChecked())
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<> dist(0, 1);
        agent.open = dist(rng);
        agent.conscientious = dist(rng);
        agent.extravert = dist(rng);
        agent.agreeable = dist(rng);
        agent.neurotic = dist(rng);
    }
}

void ConfigBuilder::assignBody(Agent& agent)
{
    agent.radius = ui->doubleSpinBox_genRadius->value();
    if (ui->radioButton_angleRand->isChecked())
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<> dist(0, 360);
        agent.angle = dist(rng);
    }
    else if (ui->radioButton_angleDir->isChecked())
        agent.angle = atan2(agent.y - ui->doubleSpinBox_angleDirY->value(), agent.x - ui->doubleSpinBox_angleDirX->value()) * 180 / M_PI - 90;
    agent.tracked = ui->checkBox_genTracked->isChecked();
}

void ConfigBuilder::assignEmotion(Agent& agent)
{
    if (ui->radioButton_emotManual->isChecked())
    {
        agent.valence = ui->doubleSpinBox_genVal->value();
        agent.arousal = ui->doubleSpinBox_genAro->value();
        agent.panic = ui->doubleSpinBox_genPanic->value();
    }
    else if (ui->radioButton_emotNorm->isChecked())
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::normal_distribution<> distVal(ui->doubleSpinBox_genVal->value(), ui->doubleSpinBox_sdVal->value());
        std::normal_distribution<> distAro(ui->doubleSpinBox_genAro->value(), ui->doubleSpinBox_sdAro->value());
        std::normal_distribution<> distPanic(ui->doubleSpinBox_genPanic->value(), ui->doubleSpinBox_sdPanic->value());
        agent.valence = distVal(rng);
        agent.arousal = distAro(rng);
        agent.panic = distPanic(rng);
    }
    else if (ui->radioButton_emotUni->isChecked())
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<> dist(-1, 1);
        agent.valence = dist(rng);
        agent.arousal = dist(rng);
        agent.panic = dist(rng);
    }
    agent.attPrefValence = ui->doubleSpinBox_genAttPrefVal->value();
    agent.attPrefArousal = ui->doubleSpinBox_genAttPrefAro->value();
}

void ConfigBuilder::updateAgentList()
{
    ui->p3_listWidgetAgents->blockSignals(true);
    ui->p3_listWidgetAgents->clear();
    ui->p3_listWidgetAgents->blockSignals(false);
    for (size_t agent = 0; agent < d_agents.size(); ++agent)
    {
        QListWidgetItem* agentItem = new QListWidgetItem(ui->p3_listWidgetAgents);
        agentItem->setText(QString("agent ") + QString::number(agent));
        ui->p3_listWidgetAgents->insertItem(agent, agentItem);
    }

    if (d_selectedAgent != -1)
        ui->p3_listWidgetAgents->setCurrentItem(ui->p3_listWidgetAgents->item(d_selectedAgent));
    else
        ui->p2_treeWidget_obstacles->clearSelection();
}

void ConfigBuilder::loadAgentInfo()
{
    if (d_selectedAgent == -1)
    {
        ui->label_agentIdx->setEnabled(false);
        ui->doubleSpinBox_O->setEnabled(false);
        ui->doubleSpinBox_C->setEnabled(false);
        ui->doubleSpinBox_E->setEnabled(false);
        ui->doubleSpinBox_A->setEnabled(false);
        ui->doubleSpinBox_N->setEnabled(false);
        ui->doubleSpinBox_agentX->setEnabled(false);
        ui->doubleSpinBox_agentY->setEnabled(false);
        ui->doubleSpinBox_angle->setEnabled(false);
        ui->doubleSpinBox_radius->setEnabled(false);
        ui->checkBox_tracked->setEnabled(false);
        ui->doubleSpinBox_val->setEnabled(false);
        ui->doubleSpinBox_aro->setEnabled(false);
        ui->doubleSpinBox_panic->setEnabled(false);
        ui->doubleSpinBox_attPrefVal->setEnabled(false);
        ui->doubleSpinBox_attPrefAro->setEnabled(false);
        ui->p3_removeSelectedAgent->setEnabled(false);
        ui->p3_saveChanges->setEnabled(0);
        ui->doubleSpinBox_O->setValue(0);
        ui->doubleSpinBox_C->setValue(0);
        ui->doubleSpinBox_E->setValue(0);
        ui->doubleSpinBox_A->setValue(0);
        ui->doubleSpinBox_N->setValue(0);
        ui->doubleSpinBox_agentX->setValue(0);
        ui->doubleSpinBox_agentY->setValue(0);
        ui->doubleSpinBox_angle->setValue(0);
        ui->doubleSpinBox_radius->setValue(0);
        ui->checkBox_tracked->setChecked(false);
        ui->doubleSpinBox_val->setValue(0);
        ui->doubleSpinBox_aro->setValue(0);
        ui->doubleSpinBox_panic->setValue(0);
        ui->doubleSpinBox_attPrefVal->setValue(0);
        ui->doubleSpinBox_attPrefAro->setValue(0);
    }
    else
    {
        if (d_general.model == "DECADE")
        {
            ui->doubleSpinBox_val->setEnabled(true);
            ui->lb3_val->setEnabled(true);
            ui->doubleSpinBox_aro->setEnabled(true);
            ui->lb3_aro->setEnabled(true);
            ui->doubleSpinBox_attPrefVal->setEnabled(true);
            ui->lb3_attPrefVal->setEnabled(true);
            ui->doubleSpinBox_attPrefAro->setEnabled(true);
            ui->lb3_attPrefAro->setEnabled(true);
            ui->doubleSpinBox_panic->setEnabled(false);
            ui->lb3_panic->setEnabled(false);

        }
        else if (d_general.model == "Durupinar")
        {
            ui->doubleSpinBox_val->setEnabled(false);
            ui->lb3_val->setEnabled(false);
            ui->doubleSpinBox_aro->setEnabled(false);
            ui->lb3_aro->setEnabled(false);
            ui->doubleSpinBox_attPrefVal->setEnabled(false);
            ui->lb3_attPrefVal->setEnabled(false);
            ui->doubleSpinBox_attPrefAro->setEnabled(false);
            ui->lb3_attPrefAro->setEnabled(false);
            ui->doubleSpinBox_panic->setEnabled(true);
            ui->lb3_panic->setEnabled(true);
        }
        ui->label_agentIdx->setEnabled(true);
        ui->doubleSpinBox_O->setEnabled(true);
        ui->doubleSpinBox_C->setEnabled(true);
        ui->doubleSpinBox_E->setEnabled(true);
        ui->doubleSpinBox_A->setEnabled(true);
        ui->doubleSpinBox_N->setEnabled(true);
        ui->doubleSpinBox_agentX->setEnabled(true);
        ui->doubleSpinBox_agentY->setEnabled(true);
        ui->doubleSpinBox_angle->setEnabled(true);
        ui->doubleSpinBox_radius->setEnabled(true);
        ui->checkBox_tracked->setEnabled(true);
        ui->p3_removeSelectedAgent->setEnabled(true);
        ui->p3_saveChanges->setEnabled(true);
        ui->doubleSpinBox_O->setValue(d_agents[d_selectedAgent].open);
        ui->doubleSpinBox_C->setValue(d_agents[d_selectedAgent].conscientious);
        ui->doubleSpinBox_E->setValue(d_agents[d_selectedAgent].extravert);
        ui->doubleSpinBox_A->setValue(d_agents[d_selectedAgent].agreeable);
        ui->doubleSpinBox_N->setValue(d_agents[d_selectedAgent].neurotic);
        ui->doubleSpinBox_agentX->setValue(d_agents[d_selectedAgent].x);
        ui->doubleSpinBox_agentY->setValue(d_agents[d_selectedAgent].y);
        ui->doubleSpinBox_angle->setValue(d_agents[d_selectedAgent].angle);
        ui->doubleSpinBox_radius->setValue(d_agents[d_selectedAgent].radius);
        ui->checkBox_tracked->setChecked(d_agents[d_selectedAgent].tracked);
        ui->doubleSpinBox_val->setValue(d_agents[d_selectedAgent].valence);
        ui->doubleSpinBox_aro->setValue(d_agents[d_selectedAgent].arousal);
        ui->doubleSpinBox_attPrefVal->setValue(d_agents[d_selectedAgent].attPrefValence);
        ui->doubleSpinBox_attPrefAro->setValue(d_agents[d_selectedAgent].attPrefArousal);
    }
    ui->label_agentIdx->setText(QString::number(d_selectedAgent));
    ui->label_totalAgents->setText(QString::number(d_agents.size()));
    if (d_general.model == "DECADE")
    {
        ui->doubleSpinBox_genVal->setEnabled(true);
        ui->doubleSpinBox_sdVal->setEnabled(true);
        ui->lb3_valGen->setEnabled(true);
        ui->lb3_sdValGen->setEnabled(true);
        ui->doubleSpinBox_genAro->setEnabled(true);
        ui->doubleSpinBox_sdAro->setEnabled(true);
        ui->lb3_aroGen->setEnabled(true);
        ui->lb3_sdAroGen->setEnabled(true);
        ui->doubleSpinBox_genAttPrefVal->setEnabled(true);
        ui->lb3_genAttPrefVal->setEnabled(true);
        ui->doubleSpinBox_genAttPrefAro->setEnabled(true);
        ui->lb3_genAttPrefAro->setEnabled(true);
        ui->doubleSpinBox_genPanic->setEnabled(false);
        ui->doubleSpinBox_sdPanic->setEnabled(false);
        ui->lb3_panicGen->setEnabled(false);
        ui->lb3_sdPanicGen->setEnabled(false);

    }
    else if (d_general.model == "Durupinar")
    {
        ui->doubleSpinBox_genVal->setEnabled(false);
        ui->doubleSpinBox_sdVal->setEnabled(false);
        ui->lb3_valGen->setEnabled(false);
        ui->lb3_sdValGen->setEnabled(false);
        ui->doubleSpinBox_genAro->setEnabled(false);
        ui->doubleSpinBox_sdAro->setEnabled(false);
        ui->lb3_aroGen->setEnabled(false);
        ui->lb3_sdAroGen->setEnabled(false);
        ui->doubleSpinBox_genAttPrefVal->setEnabled(false);
        ui->lb3_genAttPrefVal->setEnabled(false);
        ui->doubleSpinBox_genAttPrefAro->setEnabled(false);
        ui->lb3_genAttPrefAro->setEnabled(false);
        ui->doubleSpinBox_genPanic->setEnabled(true);
        ui->doubleSpinBox_sdPanic->setEnabled(true);
        ui->lb3_panicGen->setEnabled(true);
        ui->lb3_sdPanicGen->setEnabled(true);
    }
    d_glWidget->update();
}

void ConfigBuilder::appendGeneral(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* root)
{
    xml_node<>* general = doc.allocate_node(node_element, "general");
    root->append_node(general);
    xml_node<>* configName = doc.allocate_node(node_element, "configName");
    configName->value(d_general.configName.c_str());
    general->append_node(configName);
    xml_node<>* outputPath = doc.allocate_node(node_element, "outputPath");
    outputPath->value(d_general.outputPath.c_str());
    general->append_node(outputPath);
    xml_node<>* xSize = doc.allocate_node(node_element, "xSize");
    xSize->value(float2char(doc, d_general.xSize));
    general->append_node(xSize);
    xml_node<>* ySize = doc.allocate_node(node_element, "ySize");
    ySize->value(float2char(doc, d_general.ySize));
    general->append_node(ySize);
    xml_node<>* timeStep = doc.allocate_node(node_element, "timeStep");
    timeStep->value(float2char(doc, d_general.timeStep));
    general->append_node(timeStep);
    xml_node<>* endStep = doc.allocate_node(node_element, "endStep");
    endStep->value(int2char(doc, d_general.endStep));
    general->append_node(endStep);
    xml_node<>* psOuterDist = doc.allocate_node(node_element, "psOuterDist");
    psOuterDist->value(float2char(doc, d_general.psOuterDist));
    general->append_node(psOuterDist);
    xml_node<>* psCritDist = doc.allocate_node(node_element, "psCritDist");
    psCritDist->value(float2char(doc, d_general.psCritDist));
    general->append_node(psCritDist);
    xml_node<>* walkSpeed = doc.allocate_node(node_element, "walkSpeed");
    walkSpeed->value(float2char(doc, d_general.walkSpeed));
    general->append_node(walkSpeed);
    xml_node<>* maxSpeed = doc.allocate_node(node_element, "maxSpeed");
    maxSpeed->value(float2char(doc, d_general.maxSpeed));
    general->append_node(maxSpeed);
    xml_node<>* maxNeighbours = doc.allocate_node(node_element, "maxNeighbours");
    maxNeighbours->value(int2char(doc, d_general.maxNeighbours));
    general->append_node(maxNeighbours);
    xml_node<>* maxObstacles = doc.allocate_node(node_element, "maxObstacles");
    maxObstacles->value(int2char(doc, d_general.maxObstacles));
    general->append_node(maxObstacles);
    xml_node<>* perceptDistance = doc.allocate_node(node_element, "perceptDistance");
    perceptDistance->value(float2char(doc, d_general.perceptDistance));
    general->append_node(perceptDistance);
    xml_node<>* fov = doc.allocate_node(node_element, "fov");
    fov->value(float2char(doc, d_general.fov));
    general->append_node(fov);
    xml_node<>* eventStep = doc.allocate_node(node_element, "eventStep");
    eventStep->value(int2char(doc, d_general.eventStep));
    general->append_node(eventStep);
    xml_node<>* eventX = doc.allocate_node(node_element, "eventX");
    eventX->value(float2char(doc, d_general.eventX));
    general->append_node(eventX);
    xml_node<>* eventY = doc.allocate_node(node_element, "eventY");
    eventY->value(float2char(doc, d_general.eventY));
    general->append_node(eventY);
    xml_node<>* eventIntensity = doc.allocate_node(node_element, "eventIntensity");
    eventIntensity->value(float2char(doc, d_general.eventIntensity));
    general->append_node(eventIntensity);
    xml_node<>* eventDistance = doc.allocate_node(node_element, "eventDistance");
    eventDistance->value(float2char(doc, d_general.eventDistance));
    general->append_node(eventDistance);
    if (d_general.model == "DECADE")
    {
        xml_node<>* contagion = doc.allocate_node(node_element, "contagion");
        contagion->value(d_general.contagion ? "1" : "0");
        general->append_node(contagion);
        xml_node<>* densityEffect = doc.allocate_node(node_element, "densityEffect");
        densityEffect->value(d_general.densityEffect ? "1" : "0");
        general->append_node(densityEffect);
        xml_node<>* densityImpact = doc.allocate_node(node_element, "densityImpact");
        densityImpact->value(float2char(doc, d_general.densityImpact));
        general->append_node(densityImpact);
        xml_node<>* pushImpact = doc.allocate_node(node_element, "pushImpact");
        pushImpact->value(float2char(doc, d_general.pushImpact));
        general->append_node(pushImpact);
        xml_node<>* attentionBias = doc.allocate_node(node_element, "attentionBias");
        attentionBias->value(d_general.attentionBias ? "1" : "0");
        general->append_node(attentionBias);
        xml_node<>* attBiasStrength = doc.allocate_node(node_element, "attBiasStrength");
        attBiasStrength->value(float2char(doc, d_general.attBiasStrength));
        general->append_node(attBiasStrength);
        xml_node<>* beta = doc.allocate_node(node_element, "beta");
        beta->value(float2char(doc, d_general.beta));
        general->append_node(beta);
        xml_node<>* maxForce = doc.allocate_node(node_element, "maxForce");
        maxForce->value(float2char(doc, d_general.maxForce));
        general->append_node(maxForce);
        xml_node<>* minRegulationTime = doc.allocate_node(node_element, "minRegulationTime");
        minRegulationTime->value(float2char(doc, d_general.minRegulationTime));
        general->append_node(minRegulationTime);
        xml_node<>* maxRegulationTime = doc.allocate_node(node_element, "maxRegulationTime");
        maxRegulationTime->value(float2char(doc, d_general.maxRegulationTime));
        general->append_node(maxRegulationTime);
    }
    else if (d_general.model == "Durupinar")
    {
        xml_node<>* nDoses = doc.allocate_node(node_element, "nDoses");
        nDoses->value(int2char(doc, d_general.nDoses));
        general->append_node(nDoses);
        xml_node<>* meanDoseSize = doc.allocate_node(node_element, "meanDoseSize");
        meanDoseSize->value(float2char(doc, d_general.meanDoseSize));
        general->append_node(meanDoseSize);
    }
}

void ConfigBuilder::appendAgents(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* root)
{
    xml_node<>* agents = doc.allocate_node(node_element, "agents");
    root->append_node(agents);
    for (int idx = 0; idx < d_agents.size(); ++idx)
    {
        xml_node<>* agent = doc.allocate_node(node_element, agentLabel(doc, idx));
        agents->append_node(agent);
        appendPersonality(doc, agent, idx);
        appendEmotion(doc, agent, idx);
        appendBody(doc, agent, idx);
    }
}

void ConfigBuilder::appendPersonality(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* agent, size_t const idx)
{
    xml_node<>* personality = doc.allocate_node(node_element, "personality");
    agent->append_node(personality);
    xml_node<>* open = doc.allocate_node(node_element, "open");
    open->value(float2char(doc, d_agents[idx].open));
    personality->append_node(open);
    xml_node<>* conscientious = doc.allocate_node(node_element, "conscientious");
    conscientious->value(float2char(doc, d_agents[idx].conscientious));
    personality->append_node(conscientious);
    xml_node<>* extravert = doc.allocate_node(node_element, "extravert");
    extravert->value(float2char(doc, d_agents[idx].extravert));
    personality->append_node(extravert);
    xml_node<>* agreeable = doc.allocate_node(node_element, "agreeable");
    agreeable->value(float2char(doc, d_agents[idx].agreeable));
    personality->append_node(agreeable);
    xml_node<>* neurotic = doc.allocate_node(node_element, "neurotic");
    neurotic->value(float2char(doc, d_agents[idx].neurotic));
    personality->append_node(neurotic);
}

void ConfigBuilder::appendEmotion(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* agent, size_t const idx)
{
    xml_node<>* emotion = doc.allocate_node(node_element, "emotion");
    agent->append_node(emotion);
    if (d_general.model == "DECADE")
    {
        xml_node<>* valence = doc.allocate_node(node_element, "valence");
        valence->value(float2char(doc, d_agents[idx].valence));
        emotion->append_node(valence);
        xml_node<>* arousal = doc.allocate_node(node_element, "arousal");
        arousal->value(float2char(doc, d_agents[idx].arousal));
        emotion->append_node(arousal);
        xml_node<>* attPrefValence = doc.allocate_node(node_element, "attPrefValence");
        attPrefValence->value(float2char(doc, d_agents[idx].attPrefValence));
        emotion->append_node(attPrefValence);
        xml_node<>* attPrefArousal = doc.allocate_node(node_element, "attPrefArousal");
        attPrefArousal->value(float2char(doc, d_agents[idx].attPrefArousal));
        emotion->append_node(attPrefArousal);
    }
    else if (d_general.model == "Durupinar")
    {
        xml_node<>* panic = doc.allocate_node(node_element, "panic");
        panic->value(float2char(doc, d_agents[idx].panic));
        emotion->append_node(panic);
    }
}

void ConfigBuilder::appendBody(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* agent, size_t const idx)
{
    xml_node<>* body = doc.allocate_node(node_element, "body");
    agent->append_node(body);
    xml_node<>* xPos = doc.allocate_node(node_element, "xPos");
    xPos->value(float2char(doc, d_agents[idx].x));
    body->append_node(xPos);
    xml_node<>* yPos = doc.allocate_node(node_element, "yPos");
    yPos->value(float2char(doc, d_agents[idx].y));
    body->append_node(yPos);
    xml_node<>* angle = doc.allocate_node(node_element, "angle");
    angle->value(float2char(doc, d_agents[idx].angle));
    body->append_node(angle);
    xml_node<>* radius = doc.allocate_node(node_element, "radius");
    radius->value(float2char(doc, d_agents[idx].radius));
    body->append_node(radius);
    xml_node<>* tracked = doc.allocate_node(node_element, "tracked");
    tracked->value(d_agents[idx].tracked ? "1" : "0");
    body->append_node(tracked);
}

void ConfigBuilder::appendObstacles(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* root)
{
    xml_node<>* obstacles = doc.allocate_node(node_element, "obstacles");
    root->append_node(obstacles);
    for (int idx = 0; idx < d_obstacles.size(); ++idx)
    {
        xml_node<>* obstacle = doc.allocate_node(node_element, obstacleLabel(doc, idx));
        obstacles->append_node(obstacle);
        xml_node<>* vertices = doc.allocate_node(node_element, "vertices");
        obstacle->append_node(vertices);
        appendVertices(doc, vertices, idx);
        xml_node<>* fillColour = doc.allocate_node(node_element, "fillColour");
        fillColour->value(d_obstacles[idx].fillCol.c_str());
        obstacle->append_node(fillColour);
        xml_node<>* lineColour = doc.allocate_node(node_element, "lineColour");
        lineColour->value(d_obstacles[idx].lineCol.c_str());
        obstacle->append_node(lineColour);
        xml_node<>* lineWidth = doc.allocate_node(node_element, "lineWidth");
        lineWidth->value(float2char(doc, d_obstacles[idx].lineWidth));
        obstacle->append_node(lineWidth);
    }
}

void ConfigBuilder::appendVertices(rapidxml::xml_document<>& doc, rapidxml::xml_node<>* vertices, int idx)
{
    for (int vrtxIdx = 0; vrtxIdx < d_obstacles[idx].vrtcX.size(); ++vrtxIdx)
    {
        xml_node<>* vertex = doc.allocate_node(node_element, vertexLabel(doc, vrtxIdx));
        vertices->append_node(vertex);
        xml_node<>* x = doc.allocate_node(node_element, "x");
        x->value(float2char(doc, d_obstacles[idx].vrtcX[vrtxIdx]));
        vertex->append_node(x);
        xml_node<>* y = doc.allocate_node(node_element, "y");
        y->value(float2char(doc, d_obstacles[idx].vrtcY[vrtxIdx]));
        vertex->append_node(y);
        xml_node<>* nextV = doc.allocate_node(node_element, "nextV");
        nextV->value(int2char(doc, d_obstacles[idx].nextV[vrtxIdx]));
        vertex->append_node(nextV);
        xml_node<>* prevV = doc.allocate_node(node_element, "prevV");
        prevV->value(int2char(doc, d_obstacles[idx].prevV[vrtxIdx]));
        vertex->append_node(prevV);
    }
}

char* ConfigBuilder::float2char(rapidxml::xml_document<>& doc, float value)
{
    char tmpval[10];
    sprintf(tmpval, "%f", value);
    return doc.allocate_string(tmpval);
}

char* ConfigBuilder::int2char(rapidxml::xml_document<>& doc, int value)
{
    char tmpval[10];
    sprintf(tmpval, "%d", value);
    return doc.allocate_string(tmpval);
}

char* ConfigBuilder::agentLabel(rapidxml::xml_document<>& doc, int value)
{
    char tmpval[10];
    sprintf(tmpval, "a%d", value);
    return doc.allocate_string(tmpval);
}

char* ConfigBuilder::obstacleLabel(rapidxml::xml_document<>& doc, int value)
{
    char tmpval[10];
    sprintf(tmpval, "o%d", value);
    return doc.allocate_string(tmpval);
}

char* ConfigBuilder::vertexLabel(rapidxml::xml_document<>& doc, int value)
{
    char tmpval[10];
    sprintf(tmpval, "v%d", value);
    return doc.allocate_string(tmpval);
}

//[ SLOTS ]
void ConfigBuilder::p1_next()
{
    saveGeneral();
    ui->stackedWidget->setCurrentIndex(1);
    updateObstacleTree();
    loadVrtxInfo();
}

void ConfigBuilder::p1_apply()
{
    saveGeneral();
    loadGeneral();
}


void ConfigBuilder::p2_previous()
{
    loadGeneral();
    ui->stackedWidget->setCurrentIndex(0);
}

void ConfigBuilder::p2_next()
{
    ui->stackedWidget->setCurrentIndex(2);
    updateAgentList();
    loadAgentInfo();
}

void ConfigBuilder::p2_newObstacle()
{
    Obstacle tmp;
    d_obstacles.push_back(tmp);
    d_selectedObstacle = d_obstacles.size() - 1;
    d_selectedVertex = -1;
    ui->p2_treeWidget_obstacles->setCurrentItem(ui->p2_treeWidget_obstacles->topLevelItem(d_selectedObstacle));
    updateObstacleTree();
    loadVrtxInfo();
}

void ConfigBuilder::p2_removePoint()
{
    if (d_selectedVertex != -1)
    {
        Obstacle& obst = d_obstacles[d_selectedObstacle];
        obst.vrtcX.erase(obst.vrtcX.begin() + d_selectedVertex);
        obst.vrtcY.erase(obst.vrtcY.begin() + d_selectedVertex);
        obst.nextV.erase(obst.nextV.begin() + d_selectedVertex);
        obst.prevV.erase(obst.prevV.begin() + d_selectedVertex);
        for (size_t vrtx = 0; vrtx < obst.nextV.size(); ++vrtx)
        {
            if (vrtx == obst.nextV.size() - 1)
                obst.nextV[vrtx] = 0;
            else
                obst.nextV[vrtx] = vrtx + 1;
            if (vrtx == 0)
                obst.prevV[vrtx] = obst.nextV.size() - 1;
            else
                obst.prevV[vrtx] = vrtx - 1;
        }
        if (d_selectedVertex == obst.vrtcX.size())
            d_selectedVertex -= 1;
        if (d_selectedVertex != -1)
            ui->p2_treeWidget_obstacles->setCurrentItem(ui->p2_treeWidget_obstacles->topLevelItem(d_selectedObstacle)->child(d_selectedVertex));
        else
            ui->p2_treeWidget_obstacles->setCurrentItem(ui->p2_treeWidget_obstacles->topLevelItem(d_selectedObstacle));
        updateObstacleTree();
        loadVrtxInfo();
    }
}

void ConfigBuilder::p2_removeObstacle()
{
    if (d_selectedObstacle != -1)
        d_obstacles.erase(d_obstacles.begin() + d_selectedObstacle);
    if (d_selectedObstacle == d_obstacles.size())
        d_selectedObstacle -= 1;
    d_selectedVertex = -1;
    updateObstacleTree();
    loadVrtxInfo();
}

void ConfigBuilder::p2_treeItemSelected()
{
    int obstacle = -1;
    int vrtx = -1;
    if (ui->p2_treeWidget_obstacles->currentItem())
    {
        if (ui->p2_treeWidget_obstacles->currentItem()->parent())
        {
            obstacle = ui->p2_treeWidget_obstacles->currentIndex().parent().row();
            vrtx = ui->p2_treeWidget_obstacles->currentIndex().row();
            ui->p2_treeWidget_obstacles->expandItem(ui->p2_treeWidget_obstacles->currentItem()->parent());
        }
        else
        {
            obstacle = ui->p2_treeWidget_obstacles->currentIndex().row();
            ui->p2_treeWidget_obstacles->expandItem(ui->p2_treeWidget_obstacles->currentItem());
            vrtx = -1;
        }

        if (vrtx == d_obstacles[obstacle].vrtcX.size())
            vrtx -= 1;
        d_selectedObstacle = obstacle;
        d_selectedVertex = vrtx;

        loadVrtxInfo();
    }
}

void ConfigBuilder::p2_saveChanges()
{
    if (d_selectedObstacle != -1)
    {
        d_obstacles[d_selectedObstacle].fillCol = ui->lineEdit_fillCol->text().toStdString();
        d_obstacles[d_selectedObstacle].lineCol = ui->lineEdit_lineCol->text().toStdString();
        d_obstacles[d_selectedObstacle].lineWidth = ui->doubleSpinBox_lineWidth->value();
        if (d_selectedVertex != -1)
        {
            d_obstacles[d_selectedObstacle].vrtcX[d_selectedVertex] = ui->doubleSpinBox_vrtxX->value();
            d_obstacles[d_selectedObstacle].vrtcY[d_selectedVertex] = ui->doubleSpinBox_vrtxY->value();
            d_obstacles[d_selectedObstacle].nextV[d_selectedVertex] = ui->label_vrtxNext->text().toInt();
            d_obstacles[d_selectedObstacle].prevV[d_selectedVertex] = ui->label_vrtxPrevious->text().toInt();
        }
    }
    d_glWidget->update();
}


void ConfigBuilder::p3_previous()
{
    ui->stackedWidget->setCurrentIndex(1);
    loadVrtxInfo();
}

void ConfigBuilder::p3_build()
{
    QString filePath;
    filePath = QFileDialog::getSaveFileName(this, "Save config file as..",
        d_workDir, "Config Files (*.xml);;All files (*)");
    if (filePath != "")
    {
        rapidxml::xml_document<> doc;
        xml_node<>* decl = doc.allocate_node(node_declaration);
        decl->append_attribute(doc.allocate_attribute("version", "1.0"));
        decl->append_attribute(doc.allocate_attribute("encoding", "utf-8"));
        doc.append_node(decl);
        xml_node<>* root = doc.allocate_node(node_element, "config");
        root->append_attribute(doc.allocate_attribute("model", d_general.model.c_str()));
        doc.append_node(root);
        appendGeneral(doc, root);
        appendAgents(doc, root);
        appendObstacles(doc, root);
        ofstream file(filePath.toStdString());
        file << doc;
        file.close();
        doc.clear();
        QMessageBox::information(this, "Done",
            QString("Config file was build and save to ") + filePath,
            QMessageBox::Ok);

        const wstring subKey = L"SOFTWARE\\DECADE GPU";
        RegKey key{ HKEY_CURRENT_USER, subKey };
        key.Open(HKEY_CURRENT_USER, subKey);
        QFileInfo fi(filePath);
        d_workDir = fi.path();
        if (key.TrySetExpandStringValue(L"LastLocation", fi.path().toStdWString()).Failed())
            wcout << L"RegKey::TrySetLastLocation failed\n";
    }
}

void ConfigBuilder::p3_generateAgents()
{
    ui->progressBar_gen->setEnabled(true);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<> distX(ui->doubleSpinBox_xMin->value(), ui->doubleSpinBox_xMax->value());
    std::uniform_real_distribution<> distY(ui->doubleSpinBox_yMin->value(), ui->doubleSpinBox_yMax->value());
    for (size_t idx = 0; idx < ui->spinBox_genAgents->value(); ++idx)
    {
        if (idx % 100 == 0)\
        {
            ui->progressBar_gen->setValue(float(idx) / ui->spinBox_genAgents->value() * 100);
            d_glWidget->update();
            qApp->processEvents();
        }
        Agent newAgent;
        size_t const timeOut = 10000;
        size_t counter = 0;
        while (!placeAgent(newAgent, distX(rng), distY(rng)) && counter < timeOut)
            counter += 1;
        if (counter != timeOut)
        {
            assignPersonality(newAgent);
            assignBody(newAgent);
            assignEmotion(newAgent);
            d_agents.push_back(newAgent);
        }
        else
            break;
    }
    d_selectedAgent = d_agents.size() - 1;
    updateAgentList();
    loadAgentInfo();
    ui->progressBar_gen->setEnabled(false);
    ui->progressBar_gen->setValue(0);
}

void ConfigBuilder::p3_saveChanges()
{
    if (d_selectedAgent != -1)
    {
        d_agents[d_selectedAgent].open = ui->doubleSpinBox_O->value();
        d_agents[d_selectedAgent].conscientious = ui->doubleSpinBox_C->value();
        d_agents[d_selectedAgent].extravert = ui->doubleSpinBox_E->value();
        d_agents[d_selectedAgent].agreeable = ui->doubleSpinBox_A->value();
        d_agents[d_selectedAgent].neurotic = ui->doubleSpinBox_N->value();
        d_agents[d_selectedAgent].x = ui->doubleSpinBox_agentX->value();
        d_agents[d_selectedAgent].y = ui->doubleSpinBox_agentY->value();
        d_agents[d_selectedAgent].angle = ui->doubleSpinBox_angle->value();
        d_agents[d_selectedAgent].radius = ui->doubleSpinBox_radius->value();
        d_agents[d_selectedAgent].tracked = ui->checkBox_tracked->isChecked();
        d_agents[d_selectedAgent].valence = ui->doubleSpinBox_val->value();
        d_agents[d_selectedAgent].arousal = ui->doubleSpinBox_aro->value();
        d_agents[d_selectedAgent].panic = ui->doubleSpinBox_panic->value();
        d_agents[d_selectedAgent].attPrefValence = ui->doubleSpinBox_attPrefVal->value();
        d_agents[d_selectedAgent].attPrefArousal = ui->doubleSpinBox_attPrefAro->value();
    }
    d_glWidget->update();
}

void ConfigBuilder::p3_removeSelectedAgent()
{
    if (d_selectedAgent != -1)
    {
        d_agents.erase(d_agents.begin() + d_selectedAgent);
        if (d_selectedAgent == d_agents.size())
            d_selectedAgent -= 1;
        updateAgentList();
        loadAgentInfo();
    }
}

void ConfigBuilder::p3_removeAllAgents()
{
    d_agents.clear();
    d_selectedAgent = -1;
    updateAgentList();
    loadAgentInfo();
}

void ConfigBuilder::p3_listItemSelected()
{
    if (ui->p3_listWidgetAgents->currentItem())
    {
        d_selectedAgent = ui->p3_listWidgetAgents->currentIndex().row();
        loadAgentInfo();
    }
}

//[ ACCESSORS ]
size_t ConfigBuilder::page() const
{
    return ui->stackedWidget->currentIndex();
}

General* ConfigBuilder::general()
{
    return &d_general;
}

Agent* ConfigBuilder::agent(size_t idx)
{
    return &d_agents[idx];
}

int ConfigBuilder::nAgents() const
{
    return d_agents.size();
}

int ConfigBuilder::selectedAgent() const
{
    return d_selectedAgent;
}

int ConfigBuilder::selectedObstacle() const
{
    return d_selectedObstacle;
}

int ConfigBuilder::selectedVertex() const
{
    return d_selectedVertex;
}

Obstacle* ConfigBuilder::obstacle(size_t idx)
{
    return &d_obstacles[idx];
}

int ConfigBuilder::nObstacles() const
{
    return d_obstacles.size();
}


//[ MODIFIERS ]
void ConfigBuilder::selectAgent(int idx)
{
    d_selectedAgent = idx;
}

void ConfigBuilder::selectObstacle(int idx)
{
    d_selectedObstacle = idx;
}

void ConfigBuilder::selectVertex(int idx)
{
    d_selectedAgent = idx;
}