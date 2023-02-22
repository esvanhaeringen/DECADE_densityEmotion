#pragma once
#include "ConfigBuilder.h"
//#include <QGLWidget>
#include <QOpenGLWidget>
#include <QPainter>
#include <QDebug>

class ConfigBuilder;

class GlWidget : public QOpenGLWidget //QGLWidget
{
    Q_OBJECT

    ConfigBuilder* d_parent;
    QPainter d_painter;
    bool d_drawing = false;
    QImage d_map;
    float d_mapX;
    float d_mapY;

    std::map<int, double> d_pointX;
    std::map<int, double> d_pointY;
    std::map<int, double> d_angle;

    //View state
    float d_zoomFac;
    float d_zoomOffX;
    float d_zoomOffY;
    float d_sizeOffX;
    float d_sizeOffY;
    float d_cameraScale;

public:
    GlWidget(ConfigBuilder* parent);
    void resetZoom();
    void setSize();

    //[ ACCESSORS ]
    float const zoomFac() const;
    float const zoomOffX() const;
    float const zoomOffY() const;
    float const sizeOffX() const;
    float const sizeOffY() const;
    float const cameraScale() const;
    float const mapX() const;
    float const mapY() const;

    //[ MODIFIERS ]
    void setZoomFac(float value);
    void setZoomOffX(float value);
    void setZoomOffY(float value);
    void setSizeOffX(float value);
    void setSizeOffY(float value);
    void setCameraScale(float value);
    void setMapX(float value);
    void setMapY(float value);

protected:
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:

    void drawAgent(float xPos, float yPos, float radius, float angle,
        QColor cirCol, QColor triCol, bool selected, bool tracked);
    void drawEdge(float xBegin, float yBegin, float xEnd, float yEnd,
        float size, QColor lineCol);
    void drawNode(float x, float y, float radius, QColor fillCol, bool selected);
    void drawEvent(float x, float y, float dist);
    void drawMap();
    QColor outerColor(int const labelIdx);
    QColor innerColor();
};


