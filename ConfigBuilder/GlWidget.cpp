#define _USE_MATH_DEFINES

#include "GlWidget.h"
#include <QWheelEvent> 
#include <QPaintEvent>
#include <vector>
#include <cmath>
#include <QDebug>

using namespace std;

GlWidget::GlWidget(ConfigBuilder* parent)
    : d_parent(parent)
{
    setFixedSize(1000, 1000);
    setSize();
    setAutoFillBackground(true);
    d_painter.setRenderHint(QPainter::Antialiasing);
}

void GlWidget::resetZoom()
{
    d_zoomFac = 1.f;
    d_zoomOffX = 0.f;
    d_zoomOffY = 0.f;
}

void GlWidget::setSize()
{
    d_zoomFac = 1;
    d_zoomOffX = 0;
    d_zoomOffY = 0;

    d_mapX = d_parent->general()->xSize;
    d_mapY = d_parent->general()->ySize;

    d_cameraScale = 1000 / std::max(d_mapX, d_mapY);

    if (d_mapX < d_mapY)
    {
        d_sizeOffX = 500 - (d_mapX / d_mapY) * 500;
        d_sizeOffY = 0;
    }
    else if (d_mapY < d_mapX)
    {
        d_sizeOffX = 0;
        d_sizeOffY = 500 - (d_mapY / d_mapX) * 500;
    }
    else
    {
        d_sizeOffX = 0;
        d_sizeOffY = 0;
    }
}

void GlWidget::paintEvent(QPaintEvent* event)
{
    d_drawing = true;
    d_painter.begin(this);
    drawMap();
    drawEvent(d_parent->general()->eventX, d_parent->general()->eventY, d_parent->general()->eventDistance);

    d_painter.setPen(Qt::NoPen);
    for (int obstacle = 0; obstacle < d_parent->nObstacles(); ++obstacle)
    {
        Obstacle* obst = d_parent->obstacle(obstacle);
        for (int vrtx = 0; vrtx < obst->vrtcX.size() && obst->vrtcX.size() > 1; ++vrtx)
            drawEdge(obst->vrtcX[vrtx], obst->vrtcY[vrtx], obst->vrtcX[obst->nextV[vrtx]],
                obst->vrtcY[obst->nextV[vrtx]], obst->lineWidth, QColor(obst->lineCol.c_str()));
        for (int vrtx = 0; vrtx < obst->vrtcX.size(); ++vrtx)
            drawNode(obst->vrtcX[vrtx], obst->vrtcY[vrtx], obst->lineWidth, QColor(obst->fillCol.c_str()),
                obstacle == d_parent->selectedObstacle() && vrtx == d_parent->selectedVertex() && d_parent->page() == 1);
    }
    for (int idx = 0; idx < d_parent->nAgents(); ++idx)
        drawAgent(d_parent->agent(idx)->x, d_parent->agent(idx)->y,
            d_parent->agent(idx)->radius, d_parent->agent(idx)->angle,
            outerColor(0), innerColor(), 
            idx == d_parent->selectedAgent() && d_parent->page() == 2,
            d_parent->agent(idx)->tracked);
    d_painter.end();
    d_drawing = false;
}

QColor GlWidget::outerColor(int const labelIdx)
{
    QColor color("#dddddd");
    return color;
}

QColor GlWidget::innerColor()
{
    QColor color;
    color.setRgb(30, 30, 30);
    return color;
}

void GlWidget::drawAgent(float xPos, float yPos, float radius, float angle,
    QColor cirCol, QColor triCol, bool selected, bool tracked)
{
    d_painter.translate(d_cameraScale * (xPos * d_zoomFac) - d_zoomOffX + d_sizeOffX,
        d_cameraScale * (yPos * d_zoomFac) - d_zoomOffY + d_sizeOffY);
    d_painter.setPen(QPen(Qt::black, 0, Qt::SolidLine, Qt::RoundCap));
    if (selected)
    {
        d_painter.setBrush(QColor("yellow"));
        d_painter.drawEllipse(QRectF(-radius * 1.5 * d_zoomFac * d_cameraScale,
            -radius * 1.5 * d_zoomFac * d_cameraScale, 2 * radius * 1.5 * d_zoomFac * d_cameraScale,
            2 * radius * 1.5 * d_zoomFac * d_cameraScale));
    }

    d_painter.setBrush(cirCol);
    d_painter.drawEllipse(QRectF(-radius * d_zoomFac * d_cameraScale,
        -radius * d_zoomFac * d_cameraScale, 2 * radius * d_zoomFac * d_cameraScale,
        2 * radius * d_zoomFac * d_cameraScale));

    if (tracked)
        d_painter.setBrush(Qt::magenta);
    else
        d_painter.setBrush(cirCol.lighter(100 + (255 - cirCol.lightness()) / 1.5));

    d_painter.rotate(angle);
    QPolygonF triangle;
    triangle << QPointF((-(2 * radius) / 3.f) * d_zoomFac * d_cameraScale,
        ((2 * radius) / 3.f) * d_zoomFac * d_cameraScale) <<
        QPointF(((2 * radius) / 3.f) * d_zoomFac * d_cameraScale, ((2 * radius) / 3.f) *
            d_zoomFac * d_cameraScale) <<
        QPointF(0, (-(2 * radius) / 2.f) * d_zoomFac * d_cameraScale);
    d_painter.drawPolygon(triangle);  // Draw a triangle on a polygonal model
    d_painter.rotate(-angle);
    d_painter.translate(-d_cameraScale * (xPos * d_zoomFac) + d_zoomOffX - d_sizeOffX,
        -d_cameraScale * (yPos * d_zoomFac) + d_zoomOffY - d_sizeOffY);
}

void GlWidget::drawEdge(float xBegin, float yBegin, float xEnd, float yEnd,
    float size, QColor lineCol)
{
    d_painter.setPen(QPen(lineCol, size, Qt::SolidLine, Qt::RoundCap));
    d_painter.drawLine(
        xBegin * d_zoomFac * d_cameraScale - d_zoomOffX + d_sizeOffX,
        yBegin * d_zoomFac * d_cameraScale - d_zoomOffY + d_sizeOffY,
        xEnd * d_zoomFac * d_cameraScale - d_zoomOffX + d_sizeOffX,
        yEnd * d_zoomFac * d_cameraScale - d_zoomOffY + d_sizeOffY);
}

void GlWidget::drawEvent(float x, float y, float dist)
{
    d_painter.setPen(QPen(Qt::red, 2 * d_zoomFac * d_cameraScale, Qt::SolidLine, Qt::RoundCap));
    d_painter.drawLine(
        (x - max(d_mapX, d_mapY) / 200) * d_zoomFac * d_cameraScale - d_zoomOffX + d_sizeOffX,
        (y - max(d_mapX, d_mapY) / 200) * d_zoomFac * d_cameraScale - d_zoomOffY + d_sizeOffY,
        (x + max(d_mapX, d_mapY) / 200) * d_zoomFac * d_cameraScale - d_zoomOffX + d_sizeOffX,
        (y + max(d_mapX, d_mapY) / 200) * d_zoomFac * d_cameraScale - d_zoomOffY + d_sizeOffY);
    d_painter.drawLine(
        (x - max(d_mapX, d_mapY) / 200) * d_zoomFac * d_cameraScale - d_zoomOffX + d_sizeOffX,
        (y + max(d_mapX, d_mapY) / 200) * d_zoomFac * d_cameraScale - d_zoomOffY + d_sizeOffY,
        (x + max(d_mapX, d_mapY) / 200) * d_zoomFac * d_cameraScale - d_zoomOffX + d_sizeOffX,
        (y - max(d_mapX, d_mapY) / 200) * d_zoomFac * d_cameraScale - d_zoomOffY + d_sizeOffY);
    d_painter.setPen(QPen(Qt::red, 2 * d_zoomFac * d_cameraScale, Qt::DotLine, Qt::RoundCap));
    d_painter.setBrush(Qt::NoBrush);
    d_painter.drawEllipse(QRectF((x - dist) * d_zoomFac * d_cameraScale - d_zoomOffX + d_sizeOffX,
        (y - dist) * d_zoomFac * d_cameraScale - d_zoomOffY + d_sizeOffY, 2 * dist * d_zoomFac * d_cameraScale,
        2 * dist * d_zoomFac * d_cameraScale));
}

void GlWidget::drawNode(float x, float y, float radius, QColor fillCol, bool selected)
{
    d_painter.setPen(QPen(Qt::NoPen));
    d_painter.translate(d_cameraScale * (x * d_zoomFac) - d_zoomOffX + d_sizeOffX,
        d_cameraScale * (y * d_zoomFac) - d_zoomOffY + d_sizeOffY);
    if (selected)
    {
        d_painter.setBrush(QColor("magenta"));
        d_painter.drawEllipse(QRectF(-radius * 1.5 * d_zoomFac * d_cameraScale,
            -radius * 1.5 * d_zoomFac * d_cameraScale, 2 * radius * 1.5 * d_zoomFac * d_cameraScale,
            2 * radius * 1.5 * d_zoomFac * d_cameraScale));
    }
    d_painter.setBrush(fillCol);
    d_painter.drawEllipse(QRectF(-radius * d_zoomFac * d_cameraScale,
        -radius * d_zoomFac * d_cameraScale, 2 * radius * d_zoomFac * d_cameraScale,
        2 * radius * d_zoomFac * d_cameraScale));
    d_painter.translate(-d_cameraScale * (x * d_zoomFac) + d_zoomOffX - d_sizeOffX,
        -d_cameraScale * (y * d_zoomFac) + d_zoomOffY - d_sizeOffY);
}

void GlWidget::drawMap()
{
    d_painter.setPen(Qt::NoPen);
    d_painter.setBrush(Qt::black);
    d_painter.drawRect(QRect(0, 0, 1000, 1000));
    d_painter.setBrush(Qt::white);
    d_painter.drawRect(QRect(
        d_sizeOffX - d_zoomOffX,
        d_sizeOffY - d_zoomOffY,
        d_mapX * d_zoomFac * d_cameraScale,
        d_mapY * d_zoomFac * d_cameraScale));
}

void GlWidget::wheelEvent(QWheelEvent* event)
{
    float oldZoomFac = d_zoomFac;
    float numStep = (event->angleDelta().y() / 8 / 15) * WHEEL_DELTA;

    //if outside zoom limits or mouse is not inside the map
    if ((d_zoomFac == 0 && numStep < 0) ||
        (d_zoomFac == 10 && numStep > 0) ||
        event->x() < d_sizeOffX - d_zoomOffX ||
        event->x() > d_sizeOffX - d_zoomOffX + d_mapX * d_zoomFac * d_cameraScale ||
        event->y() < d_sizeOffY - d_zoomOffY ||
        event->y() > d_sizeOffY - d_zoomOffY + d_mapY * d_zoomFac * d_cameraScale)
        return;

    d_zoomFac += (numStep * 0.01f);

    if (d_zoomFac < 1.f)
        d_zoomFac = 1.f;
    else if (d_zoomFac > 10.f)
        d_zoomFac = 10.f;

    float pxWidth = d_mapX * d_cameraScale;
    float pxHeight = d_mapY * d_cameraScale;
    float relPosX = d_zoomOffX / (pxWidth * oldZoomFac) +
        (event->x() - d_sizeOffX) / (pxWidth * oldZoomFac);
    float relPosY = d_zoomOffY / (pxHeight * oldZoomFac) +
        (event->y() - d_sizeOffY) / (pxHeight * oldZoomFac);
    d_zoomOffX = relPosX * (d_zoomFac * pxWidth - pxWidth);
    d_zoomOffY = relPosY * (d_zoomFac * pxHeight - pxHeight);
    update();
}

void GlWidget::mousePressEvent(QMouseEvent* event)
{
    d_parent->handleClick(
        (d_zoomOffX - d_sizeOffX + event->x()) / (d_zoomFac * d_cameraScale),
        (d_zoomOffY - d_sizeOffY + event->y()) / (d_zoomFac * d_cameraScale));
}

//[ ACCESSORS ]
float const GlWidget::zoomFac() const
{
    return d_zoomFac;
}
float const GlWidget::zoomOffX() const
{
    return d_zoomOffX;
}
float const GlWidget::zoomOffY() const
{
    return d_zoomOffY;
}
float const GlWidget::sizeOffX() const
{
    return d_sizeOffX;
}
float const GlWidget::sizeOffY() const
{
    return d_sizeOffY;
}
float const GlWidget::cameraScale() const
{
    return d_cameraScale;
}
float const GlWidget::mapX() const
{
    return d_mapX;
}
float const GlWidget::mapY() const
{
    return d_mapY;
}

//[ MODIFIERS ]
void GlWidget::setZoomFac(float value)
{
    d_zoomFac = value;
}
void GlWidget::setZoomOffX(float value)
{
    d_zoomOffX = value;
}
void GlWidget::setZoomOffY(float value)
{
    d_zoomOffY = value;
}
void GlWidget::setSizeOffX(float value)
{
    d_sizeOffX = value;
}
void GlWidget::setSizeOffY(float value)
{
    d_sizeOffY = value;
}
void GlWidget::setCameraScale(float value)
{
    d_cameraScale = value;
}
void GlWidget::setMapX(float value)
{
    d_mapX = value;
}
void GlWidget::setMapY(float value)
{
    d_mapY = value;
}