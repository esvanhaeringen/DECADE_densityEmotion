#include "monitorWidget.h"
#include <QHeaderView>

MonitorWidget::MonitorWidget(QArrow2D* parent)
    : d_parent(parent)
{
    this->setRowCount(d_nItems);
    this->setColumnCount(2);
    QStringList colHeaders;
    colHeaders << "Property" << "Value";
    this->setHorizontalHeaderLabels(colHeaders);
    this->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
    this->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    this->setEditTriggers(QAbstractItemView::NoEditTriggers);
    this->setFocusPolicy(Qt::NoFocus);
    this->setSelectionMode(QAbstractItemView::NoSelection);
    this->verticalHeader()->hide();
}

void MonitorWidget::setup()
{
    if (d_parent->model() != nullptr)
    {
        d_nItems = 13;
        this->setRowCount(d_nItems);
        for (int i = 0; i < d_nItems; i++)
        {
            this->setItem(i, 0, new QTableWidgetItem);
            this->setItem(i, 1, new QTableWidgetItem);
        }
        this->item(0, 0)->setText("ID");
        this->item(1, 0)->setText("x");
        this->item(2, 0)->setText("y");
        this->item(3, 0)->setText("Speed");
        this->item(4, 0)->setText("Angle");
        this->item(5, 0)->setText("Incoming force");
        this->item(6, 0)->setText("Density (agents/M2)");
        this->item(7, 0)->setText("Panic");
        this->item(8, 0)->setText("CombinedDose");
        this->item(9, 0)->setText("Infected");
        this->item(10, 0)->setText("Expressivity");
        this->item(11, 0)->setText("Susceptibility");
        this->item(12, 0)->setText("Regulation effect");
    }
}

void MonitorWidget::update()
{
    if (d_agent != -1)
    {
        if (this->rowCount() == 0)
            setup();
        this->item(0, 1)->setText(QString::number(d_agent));
        this->item(1, 1)->setText(QString::number(d_parent->model()->x(d_agent)));
        this->item(2, 1)->setText(QString::number(d_parent->model()->y(d_agent)));
        this->item(3, 1)->setText(QString::number(d_parent->model()->speed(d_agent)));
        this->item(4, 1)->setText(QString::number(d_parent->model()->angle(d_agent)));
        this->item(5, 1)->setText(QString::number(d_parent->model()->incomingForce(d_agent)));
        this->item(6, 1)->setText(QString::number(d_parent->model()->densityM2(d_agent)));
        this->item(7, 1)->setText(QString::number(d_parent->model()->panic(d_agent)));
        this->item(8, 1)->setText(QString::number(d_parent->model()->combinedDose(d_agent)));
        this->item(9, 1)->setText(QString::number(d_parent->model()->infected(d_agent)));
        this->item(10, 1)->setText(QString::number(d_parent->model()->expressivity(d_agent)));
        this->item(11, 1)->setText(QString::number(d_parent->model()->susceptibility(d_agent)));
        this->item(12, 1)->setText(QString::number(d_parent->model()->regulationEfficiency(d_agent)));
    }
}

void MonitorWidget::clean()
{
    d_agent = -1;
    this->setRowCount(0);
}

//[ ACCESSORS ]
int const MonitorWidget::agent() const
{
    return d_agent;
}

//[ MODIFIERS ]
void MonitorWidget::setAgent(int agent)
{
    d_agent = agent;
}

