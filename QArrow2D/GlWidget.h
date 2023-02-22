#pragma once
#include "QArrow2D.h"
#include <QGLWidget>
#include <QPainter>
#include <QDebug>

class QArrow2D;

class GlWidget : public QGLWidget
{
    Q_OBJECT

    QArrow2D* d_parent;
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
    GlWidget(QArrow2D* parent);
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
    float const capture() const;

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
        QColor cirCol, QColor triCol);
    void drawSelectedAgent(float xPos, float yPos, float radius);
    void drawSelectedNeighbour(float xPos, float yPos, float radius);
    void drawEdge(float xBegin, float yBegin, float xEnd, float yEnd,
        float size, QColor lineCol);
    void drawMap();
    QColor outerColor(size_t const idx);
    QColor innerColor(size_t const idx);
};


