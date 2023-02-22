#pragma once
#include <QTableWidget>
#include "QArrow2D.h"

class Agent;
class QArrow2D;

class MonitorWidget : public QTableWidget
{
    int d_nItems = 0;
    int d_agent = -1;
    QArrow2D* d_parent;

public:
    MonitorWidget(QArrow2D* parent);
    void setup();
    void update();
    void clean();

    //[ ACCESSORS ]
    int const agent() const;

    //[ MODIFIERS ]
    void setAgent(int agent);
};