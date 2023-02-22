#ifndef __mathFunCuda__
#define __mathFunCuda__

inline __device__ float prod(float compX1, float compY1, float compX2, float compY2)
{
    return compX1 * compX2 + compY1 * compY2;
}

inline __device__ float absSq(float compX, float compY)
{
    return prod(compX, compY, compX, compY);
}

inline __device__ float sqr(float comp)
{
    return comp * comp;
}

inline __device__ float abs(float compX, float compY)
{
    return sqrt(prod(compX, compY, compX, compY));
}

inline __device__ float det(float compX1, float compY1, float compX2, float compY2)
{
    return compX1 * compY2 - compY1 * compX2;
}

inline __device__ float norm(float comp1, float comp2)
{
    return comp1 / abs(comp1, comp2);
}

inline __device__ float distSqPointLineSegment(float compX1, float compY1,
    float compX2, float compY2, float compX3, float compY3)
{
    const float r = prod(compX3 - compX1, compY3 - compY1, compX2 - compX1,
        compY2 - compY1) / absSq(compX2 - compX1, compY2 - compY1);

    if (r < 0.0f)
        return absSq(compX3 - compX1, compY3 - compY1);
    else if (r > 1.0f)
        return absSq(compX3 - compX2, compY3 - compY2);
    else
        return absSq(compX3 - (compX1 + r * (compX2 - compX1)),
            compY3 - (compY1 + r * (compY2 - compY1)));
}

inline __device__ float leftOf(float compX1, float compY1, float compX2, float compY2,
    float compX3, float compY3)
{
    return det(compX1 - compX3, compY1 - compY3, compX2 - compX1,
        compY2 - compY1);
}

inline __device__ float inf()
{
    return __int_as_float(0x7f800000);
}

inline __device__ float euclSqrDistance(float compA1, float compB1, float compA2, float compB2)
{
    return powf((compA1 - compA2), 2) + powf((compB1 - compB2), 2);
}

inline __device__ float euclDistance(float compA1, float compB1, float compA2, float compB2)
{
    return sqrtf(powf((compA1 - compA2), 2) + powf((compB1 - compB2), 2));
}

//https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
inline __device__ bool ccw(float x1, float y1, float x2, float y2, float x3, float y3)
{
    return (y3 - y1) * (x2 - x1) > (y2 - y1) * (x3 - x1);
}

inline __device__ bool intersect(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4)
{
    return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) &&
        ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4);
}

inline __device__ bool angleInRange(int angle, int lowerLim, int upperLim)
{
    lowerLim = (lowerLim + 360) % 360;
    upperLim = (upperLim + 360) % 360;
    angle = (angle + 360) % 360;
    if (lowerLim <= upperLim)
        return angle >= lowerLim && angle <= upperLim;
    else
        return angle >= lowerLim || angle <= upperLim;
}


#endif