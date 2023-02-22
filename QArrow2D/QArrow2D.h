#pragma once
#include "GlWidget.h"
#include "MonitorWidget.h"
#include "../ModelInterface/ModelInterface.h"
#include <QMainWindow>
#include <QWidget>
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <qdatetime.h>

class GlWidget;
class MonitorWidget;

namespace Ui {
	class QArrow2D;
}

class QArrow2D: public QMainWindow
{
	Q_OBJECT
    ModelInterface* d_model = nullptr;
    bool d_ready = false;
    bool d_running = false;
    bool d_recording = false;
    std::string d_recPath;
    std::string d_config;
    GlWidget* d_glWidget;
    MonitorWidget* d_monitor;
    QTimer* d_timer;
    QTime d_time;

public:
    QArrow2D(QWidget* parent = 0);
    ~QArrow2D();

    //[ ACCESSORS ]
    ModelInterface* model();
    MonitorWidget* monitor();
    GlWidget* glWidget();
    bool ready();

public slots:
    void playPause();
    void record();
    void setSpeed(int value);
    void loadScene();
    void defaultScene();
    void openConfig();

private:
    Ui::QArrow2D* ui;
    void update();
    void setStates();
    void keyPressEvent(QKeyEvent* pe);
    void setupModel(std::string file);
    void setupCharts();
    void saveImage();
};

