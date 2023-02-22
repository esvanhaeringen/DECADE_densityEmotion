#define _USE_MATH_DEFINES

#include "GlWidget.h"
#include <QWheelEvent> 
#include <QPaintEvent>
#include <vector>
#include <cmath>
#include <QDebug>

using namespace std;

GlWidget::GlWidget(QArrow2D* parent)
    : d_parent(parent)
{
    setFixedSize(1000, 1000);
    setSize();
    setAutoFillBackground(true);
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

    if (d_parent->model() != nullptr)
    {
        d_mapX = d_parent->model()->xSize();
        d_mapY = d_parent->model()->ySize();
    }
    else
    {
        d_mapX = 1000;
        d_mapY = 1000;
    }

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
    if (d_parent->model() != nullptr && d_parent->model()->ready() && !d_drawing)
    {
        d_drawing = true;
        d_painter.begin(this);
        d_painter.setRenderHint(QPainter::Antialiasing);
        drawMap();

        for (int obstacle = 0; obstacle < d_parent->model()->nObstacles(); ++obstacle)
        {
            Obstacle* obst = d_parent->model()->obstacle(obstacle);
            for (int vrtx = 0; vrtx < obst->vrtcX.size() && obst->vrtcX.size() > 1; ++vrtx)
                drawEdge(obst->vrtcX[vrtx], obst->vrtcY[vrtx], obst->vrtcX[obst->nextV[vrtx]],
                    obst->vrtcY[obst->nextV[vrtx]], obst->lineWidth, QColor(obst->lineCol.c_str()));
        }

        //draw agents
        size_t const selected = d_parent->monitor()->agent();
        if (selected != -1)
            if (d_parent->model()->insideMap(selected))
            {
                drawSelectedAgent(d_parent->model()->x(selected), d_parent->model()->y(selected),
                    d_parent->model()->radius(selected));
                for (int idx = 0; idx < d_parent->model()->nNeighbours(selected); ++idx)
                {
                    size_t const neighbour = d_parent->model()->neighbour(selected, idx);
                    drawSelectedNeighbour(d_parent->model()->x(neighbour), d_parent->model()->y(neighbour),
                        d_parent->model()->radius(neighbour));
                }
            }

        for (int idx = 0; idx < d_parent->model()->nAgents(); ++idx)
            if (d_parent->model()->insideMap(idx))
                drawAgent(d_parent->model()->x(idx), d_parent->model()->y(idx),
                    d_parent->model()->radius(idx), d_parent->model()->angle(idx), 
                    outerColor(idx), innerColor(idx));

        d_painter.end();
        d_drawing = false;
    }
}

QColor GlWidget::outerColor(size_t const idx)
{
    return QColor(102 + 153.f * d_parent->model()->panic(idx), 
        102 * (1.f - d_parent->model()->panic(idx)),
        102 * (1.f - d_parent->model()->panic(idx)));
}

QColor GlWidget::innerColor(size_t const idx)
{
    QColor color;
    if (d_parent->model()->test(idx) == 1)
        color.setRgb(30, 30, 30 + 220);
    else if (d_parent->model()->test(idx) == 2)
        color.setRgb(30, 30+220, 30);
    else
        color.setRgb(180, 180, 180);
    return color;
}

void GlWidget::drawSelectedAgent(float xPos, float yPos, float radius)
{
    d_painter.translate(d_cameraScale * (xPos * d_zoomFac) - d_zoomOffX + d_sizeOffX,
        d_cameraScale * (yPos * d_zoomFac) - d_zoomOffY + d_sizeOffY);
    d_painter.setBrush(Qt::NoBrush);
    d_painter.setPen(QColor("red"));
    d_painter.drawEllipse(QRectF(-2.91 * d_zoomFac * d_cameraScale,
        -2.91 * d_zoomFac * d_cameraScale, 2 * 2.91 * d_zoomFac * d_cameraScale,
        2 * 2.91 * d_zoomFac * d_cameraScale));
    d_painter.setPen(Qt::NoPen);
    d_painter.setBrush(QColor("magenta"));
    d_painter.drawEllipse(QRectF(-radius * 1.2 * d_zoomFac * d_cameraScale,
        -radius * 1.2 * d_zoomFac * d_cameraScale, 2 * radius * 1.2 * d_zoomFac * d_cameraScale,
        2 * radius * 1.2 * d_zoomFac * d_cameraScale));
    d_painter.translate(-d_cameraScale * (xPos * d_zoomFac) + d_zoomOffX - d_sizeOffX,
        -d_cameraScale * (yPos * d_zoomFac) + d_zoomOffY - d_sizeOffY);
}

void GlWidget::drawSelectedNeighbour(float xPos, float yPos, float radius)
{
    d_painter.translate(d_cameraScale * (xPos * d_zoomFac) - d_zoomOffX + d_sizeOffX,
        d_cameraScale * (yPos * d_zoomFac) - d_zoomOffY + d_sizeOffY);
    d_painter.setPen(Qt::NoPen);
    d_painter.setBrush(QColor("blue"));
    d_painter.drawEllipse(QRectF(-radius * 1.2 * d_zoomFac * d_cameraScale,
        -radius * 1.2 * d_zoomFac * d_cameraScale, 2 * radius * 1.2 * d_zoomFac * d_cameraScale,
        2 * radius * 1.2 * d_zoomFac * d_cameraScale));
    d_painter.translate(-d_cameraScale * (xPos * d_zoomFac) + d_zoomOffX - d_sizeOffX,
        -d_cameraScale * (yPos * d_zoomFac) + d_zoomOffY - d_sizeOffY);
}

void GlWidget::drawAgent(float xPos, float yPos, float radius, float angle,
    QColor cirCol, QColor triCol)
{
    d_painter.translate(d_cameraScale * (xPos * d_zoomFac) - d_zoomOffX + d_sizeOffX,
        d_cameraScale * (yPos * d_zoomFac) - d_zoomOffY + d_sizeOffY);
    d_painter.setPen(QPen(Qt::black, 0, Qt::SolidLine, Qt::RoundCap));
    d_painter.setBrush(cirCol);
    d_painter.drawEllipse(QRectF(-radius * d_zoomFac * d_cameraScale,
        -radius * d_zoomFac * d_cameraScale, 2 * radius * d_zoomFac * d_cameraScale,
        2 * radius * d_zoomFac * d_cameraScale));

    d_painter.setBrush(triCol);
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
    if (d_parent->model() != nullptr)
    {
        float oldZoomFac = d_zoomFac;
        float numStep = (event->angleDelta().y() / 8 / 15) * WHEEL_DELTA;

        //if outside zoom limits or mouse is not inside the map
        if ((d_zoomFac == 0 && numStep < 0) ||
            (d_zoomFac == 10 && numStep > 0) ||
            event->position().x() < d_sizeOffX - d_zoomOffX ||
            event->position().x() > d_sizeOffX - d_zoomOffX + d_mapX * d_zoomFac * d_cameraScale ||
            event->position().y() < d_sizeOffY - d_zoomOffY ||
            event->position().y() > d_sizeOffY - d_zoomOffY + d_mapY * d_zoomFac * d_cameraScale)
            return;

        d_zoomFac += (numStep * 0.01f);

        if (d_zoomFac < 1.f)
            d_zoomFac = 1.f;
        else if (d_zoomFac > 10.f)
            d_zoomFac = 10.f;

        float pxWidth = d_mapX * d_cameraScale;
        float pxHeight = d_mapY * d_cameraScale;
        float relPosX = d_zoomOffX / (pxWidth * oldZoomFac) + 
            (event->position().x() - d_sizeOffX) / (pxWidth * oldZoomFac);
        float relPosY = d_zoomOffY / (pxHeight * oldZoomFac) + 
            (event->position().y() - d_sizeOffY) / (pxHeight * oldZoomFac);
        d_zoomOffX = relPosX * (d_zoomFac * pxWidth - pxWidth);
        d_zoomOffY = relPosY * (d_zoomFac * pxHeight - pxHeight);

    }
}

void GlWidget::mousePressEvent(QMouseEvent* event)
{
    if (d_parent->model() != nullptr)
    {
        float x = (d_zoomOffX - d_sizeOffX + event->x()) / (d_zoomFac * d_cameraScale);
        float y = (d_zoomOffY - d_sizeOffY + event->y()) / (d_zoomFac * d_cameraScale);

        int selectedAgent = -1;
        float selectedDist = pow(min(d_mapX, d_mapY) / 50, 2); //to avoid sqrt

        for (int idx = 0; idx < d_parent->model()->nAgents(); ++idx)
        {
            float dist = pow(d_parent->model()->x(idx) - x, 2) + 
                pow(d_parent->model()->y(idx) - y, 2);
            if (dist < selectedDist)
            {
                selectedAgent = idx;
                selectedDist = dist;
            }
        }
        if (selectedAgent != -1)
            d_parent->monitor()->setAgent(selectedAgent);
        else
            d_parent->monitor()->clean();
    }
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