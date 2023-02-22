#include "BaseModel.h"
#include "mathFuncCuda.cuh"
#include "math_constants.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

//CONSTANTS
#define DirX            0
#define DirY            1
#define PointX          2
#define PointY          3
#define RVO_EPSILON     0.00001f

//CUDA FUNCTION DEFINITIONS
//Kernels
__global__ void computePreference(int* nAgents, float* x, float* y, float* radius,
    float* prefVeloX, float* prefVeloY, float* prefSpeed, float* xSize, float* ySize,
    float* eventX, float* eventY, int* maxNeighbours, int* nNeighbours, 
    int* neighbours, float* distNeighbours, float* walkSpeed, float* maxSpeed, 
    float* timeStep, int* maxObstacles, int* nObstacleEdges, int* obstaclesEdges, 
    float* vrtxPointX, float* vrtxPointY, int* nextVrtcs, float* fov, float* test);
__global__ void computeVelocity(int* n, int* maxObstacles, int* maxNeigh,
    float* timeStep, float* x, float* y, float* radius, float* veloX,
    float* veloY, float* prefVeloX, float* prefVeloY, float* prefSpeed,
    float* vrtxPointX, float* vrtxPointY, float* unitDirX, float* unitDirY,
    int* prevVrtcs, int* nextVrtcs, bool* isConvex, float* timeHorizonObst,
    int* obstacles, int* nObstacles, float* timeHorizonAgnt, int* neighbours,
    int* nNeighbours, float* orcaLines, float* projLines, int* nLines,
    float* newVeloX, float* newVeloY, float* prefPersDist);
__global__ void moveAgents(int* nAgents, float* timeStep, float* x, float* y,
    float* veloX, float* veloY, float* speed, float* newVeloX, float* newVeloY,
    float* prefVeloX, float* prefVeloY, float* angle, float* radius, float* xSize,
    float* ySize, bool* insideMap, float* eventX, float* eventY);

//Device functions
__device__ bool obstacleLine(int self, int firstDirX, int firstDirY,
    int firstPointX, int firstPointY, int vrtx, float invTimeHorizonObst,
    int* n, float* x, float* y, float* radius, float* veloX, float* veloY,
    float* vrtxPointX, float* vrtxPointY, float* unitDirX, float* unitDirY,
    int* prevVrtcs, int* nextVrtcs, bool* isConvex, float* orcaLines,
    int* nLines);
__device__ void agentLine(int self, int firstDirX, int firstDirY,
    int firstPointX, int firstPointY, int other, float* timeStep,
    float* posX, float* posY, float* radius, float* veloX, float* veloY,
    float invTimeHorizon, float* orcaLines, int* numLine, float* prefPersDist);
__device__ bool linearProgram1(int self, int firstDirX, int firstDirY,
    int firstPointX, int firstPointY, int lineIdx, float radius,
    float prefVeloX, float prefVeloY, bool optimizeDirection, float* orcaLines,
    float* newVeloX, float* newVeloY);
__device__ int linearProgram2(int self, int firstDirX, int firstDirY,
    int firstPointX, int firstPointY, float radius, float prefVeloX,
    float prefVeloY, bool optimizeDirection, float* orcaLines, int numLine,
    float* newVeloX, float* newVeloY);
__device__ void linearProgram3(int self, int firstDirX, int firstDirY,
    int firstPointX, int firstPointY, int numObstLines, int beginLine,
    float radius, float* orcaLines, float* projLines, int numLine,
    float* newVeloX, float* newVeloY);

void BaseModel::updateBehaviour()
{
    int blockSize = d_blocksGPU * 32;
    int numBlocks = (d_nAgents + blockSize - 1) / blockSize;

    computePreference <<< numBlocks, blockSize >>> (
        c_nAgents, c_x, c_y, c_radius, c_prefVeloX, c_prefVeloY, c_prefSpeed, c_xSize, c_ySize,
        c_eventX, c_eventY, c_maxNeighbours, c_nNeighbours, c_neighbours, c_distNeighbours, c_walkSpeed, 
        c_maxSpeed, c_timeStep, c_maxObstacles, c_nObstacleEdges, c_obstacleEdges, c_vrtxPointX, 
        c_vrtxPointY, c_nextVrtcs, c_fov, c_test);
    cudaDeviceSynchronize();

    computeVelocity <<< numBlocks, blockSize >>> (
        c_nAgents, c_maxObstacles, c_maxNeighbours, c_timeStep, c_x, c_y, c_radius, c_veloX, c_veloY, 
        c_prefVeloX, c_prefVeloY, c_prefSpeed, c_vrtxPointX, c_vrtxPointY, c_unitDirX,
        c_unitDirY, c_prevVrtcs, c_nextVrtcs, c_isConvex, c_timeHorizonObst, c_obstacleEdges, 
        c_nObstacleEdges, c_timeHorizonAgnt, c_neighbours, c_nNeighbours, c_orcaLines, c_projLines, 
        c_nLines, c_newVeloX, c_newVeloY, c_prefPersDist);
    cudaDeviceSynchronize();

    moveAgents <<< numBlocks, blockSize >>> (
        c_nAgents, c_timeStep, c_x, c_y, c_veloX, c_veloY, c_speed, c_newVeloX, c_newVeloY, c_prefVeloX,
        c_prefVeloY, c_angle, c_radius, c_xSize, c_ySize, c_insideMap, c_eventX, c_eventY);
    cudaDeviceSynchronize();
}

__global__ void computePreference(int* nAgents, float* x, float* y, float* radius, float* prefVeloX,
    float* prefVeloY, float* prefSpeed, float* xSize, float* ySize, float* eventX, float* eventY, 
    int* maxNeighbours, int* nNeighbours, int* neighbours, float* distNeighbours, float* walkSpeed, 
    float* maxSpeed, float* timeStep, int* maxObstacles, int* nObstacleEdges, int* obstacleEdges, 
    float* vrtxPointX, float* vrtxPointY, int* nextVrtcs, float* fov, float* test)
{
    size_t const self = blockIdx.x * blockDim.x + threadIdx.x;
    if (self >= nAgents[0])
        return;
    size_t const firstNeigh = self * maxNeighbours[0];
    size_t const firstObst = self * maxObstacles[0];

    if (prefSpeed[self] > 0)
    {
        float const antiEventAngle = atan2(eventY[0] - y[self], eventX[0] - x[self]) * 180 / CUDART_PI_F + 90;
        size_t leftBin = 0, centreBin = 0, rightBin = 0;
        for (size_t idx = 0; idx < nNeighbours[self]; ++idx)
        {
            size_t const other = neighbours[firstNeigh + idx];
            float angle2Neighbour = atan2(y[other] - y[self], x[other] - x[self]) * 180 / CUDART_PI_F - 90;
            leftBin += angleInRange(angle2Neighbour, antiEventAngle - fov[0] / 2, antiEventAngle - fov[0] / 6);
            centreBin += angleInRange(angle2Neighbour, antiEventAngle - fov[0] / 6, antiEventAngle + fov[0] / 6);
            rightBin += angleInRange(angle2Neighbour, antiEventAngle + fov[0] / 6, antiEventAngle + fov[0] / 2);
        }

        for (int idx = 0; idx < nObstacleEdges[self]; ++idx)
        {
            size_t const v1 = obstacleEdges[firstObst + idx];
            size_t const v2 = nextVrtcs[v1];
            float propX = cos((antiEventAngle - fov[0] / 3 + 90) * CUDART_PI_F / 180) * 15.45;
            float propY = sin((antiEventAngle - fov[0] / 3 + 90) * CUDART_PI_F / 180) * 15.45;
            if (intersect(x[self], y[self], x[self] + propX, y[self] + propY, vrtxPointX[v1], vrtxPointY[v1], vrtxPointX[v2], vrtxPointY[v2]))
                leftBin += maxNeighbours[0];
            propX = cos((antiEventAngle + 90) * CUDART_PI_F / 180) * 15.45;
            propY = sin((antiEventAngle + 90) * CUDART_PI_F / 180) * 15.45;
            if (intersect(x[self], y[self], x[self] + propX, y[self] + propY, vrtxPointX[v1], vrtxPointY[v1], vrtxPointX[v2], vrtxPointY[v2]))
                centreBin += maxNeighbours[0];
            propX = cos((antiEventAngle + fov[0] / 3 + 90) * CUDART_PI_F / 180) * 15.45;
            propY = sin((antiEventAngle + fov[0] / 3 + 90) * CUDART_PI_F / 180) * 15.45;
            if (intersect(x[self], y[self], x[self] + propX, y[self] + propY, vrtxPointX[v1], vrtxPointY[v1], vrtxPointX[v2], vrtxPointY[v2]))
                rightBin += maxNeighbours[0];
        }

        float prefAngle = antiEventAngle;
        if (leftBin < centreBin && leftBin < rightBin)
            prefAngle -= fov[0] / 3;
        else if (rightBin < centreBin && rightBin < leftBin)
            prefAngle += fov[0] / 3;

        prefVeloX[self] = cos((prefAngle + 90) * CUDART_PI_F / 180) * prefSpeed[self];
        prefVeloY[self] = sin((prefAngle + 90) * CUDART_PI_F / 180) * prefSpeed[self];
    }
    else
    {
        prefVeloX[self] = 0;
        prefVeloY[self] = 0;
    }
}

__global__ void computeVelocity(
    int* nAgents, int* maxObstacles, int* maxNeigh, float* timeStep, float* x, float* y, float* radius,
    float* veloX, float* veloY, float* prefVeloX, float* prefVeloY, float* prefSpeed,
    float* vrtxPointX, float* vrtxPointY, float* unitDirX, float* unitDirY, int* prevVrtcs,
    int* nextVrtcs, bool* isConvex, float* timeHorizonObst, int* obstacles, int* nObstacles,
    float* timeHorizonAgnt, int* neighbours, int* nNeighbours, float* orcaLines, float* projLines,
    int* nLines, float* newVeloX, float* newVeloY, float* prefPersDist)
{
    //focal agent for current thread
    int self = blockIdx.x * blockDim.x + threadIdx.x;
    if (self >= nAgents[0])
        return;
    //indices
    int fistObst = self * maxObstacles[0];
    int firstNeigh = self * maxNeigh[0];
    int const firstDirX = DirX * nAgents[0] * (maxObstacles[0] + maxNeigh[0]) + self *
        (maxObstacles[0] + maxNeigh[0]);
    int const firstDirY = DirY * nAgents[0] * (maxObstacles[0] + maxNeigh[0]) + self *
        (maxObstacles[0] + maxNeigh[0]);
    int const firstPointX = PointX * nAgents[0] * (maxObstacles[0] + maxNeigh[0]) + self *
        (maxObstacles[0] + maxNeigh[0]);
    int const firstPointY = PointY * nAgents[0] * (maxObstacles[0] + maxNeigh[0]) + self *
        (maxObstacles[0] + maxNeigh[0]);

    nLines[self] = 0;                 //reset
    int numObstLines = 0;

    //compute orca lines for near objects
    for (int idx = 0; idx < nObstacles[self]; ++idx)
        if (obstacleLine(self, firstDirX, firstDirY, firstPointX, firstPointY,
            obstacles[fistObst + idx], 1.f / timeHorizonObst[0], nAgents, x, y,
            radius, veloX, veloY, vrtxPointX, vrtxPointY, unitDirX, unitDirY,
            prevVrtcs, nextVrtcs, isConvex, orcaLines, nLines))
        {
            projLines[firstDirX + numObstLines] = orcaLines[firstDirX +
                nLines[self]];
            projLines[firstDirY + numObstLines] = orcaLines[firstDirY +
                nLines[self]];
            projLines[firstPointX + numObstLines] = orcaLines[firstPointX +
                nLines[self]];
            projLines[firstPointY + numObstLines] = orcaLines[firstPointY +
                nLines[self]];
            ++numObstLines;
            ++nLines[self];
        }

    //compute orca lines for near agents
    for (int idx = 0; idx < nNeighbours[self]; ++idx)
        agentLine(self, firstDirX, firstDirY, firstPointX, firstPointY,
            neighbours[firstNeigh + idx], timeStep, x, y, radius, veloX, veloY,
            1.f / timeHorizonAgnt[0], orcaLines, nLines, prefPersDist);

    int lineFail = linearProgram2(self, firstDirX, firstDirY, firstPointX,
        firstPointY, prefSpeed[self], prefVeloX[self], prefVeloY[self], false,
        orcaLines, nLines[self], newVeloX, newVeloY);

    if (lineFail < nLines[self])
        linearProgram3(self, firstDirX, firstDirY, firstPointX, firstPointY,
            numObstLines, lineFail, prefSpeed[self], orcaLines, projLines,
            nLines[self], newVeloX, newVeloY);
}

__global__ void moveAgents(
    int* nAgents, float* timeStep, float* x, float* y, float* veloX, float* veloY, float* speed,
    float* newVeloX, float* newVeloY, float* prefVeloX, float* prefVeloY, float* angle, float* radius, 
    float* xSize, float* ySize, bool* insideMap, float* eventX, float* eventY)
{
    int self = blockIdx.x * blockDim.x + threadIdx.x;
    if (self >= nAgents[0])
        return;

    veloX[self] = newVeloX[self];
    veloY[self] = newVeloY[self];
    speed[self] = abs(veloX[self], veloY[self]);
    
    if (speed[self] > 0)
        angle[self] = atan2(veloY[self], veloX[self]) * 180 / CUDART_PI_F + 90;
    else
        angle[self] = atan2(y[self] - eventY[0], x[self] - eventX[0]) * 180 / CUDART_PI_F - 90;;
    x[self] += veloX[self];
    y[self] += veloY[self];
    if (x[self] < -radius[self] || x[self] > xSize[0] + radius[self] ||
        y[self] < -radius[self] || y[self] > ySize[0] + radius[self])
        insideMap[self] = false;
    else
        insideMap[self] = true;
}

__device__ bool obstacleLine(
    int self, int firstDirX, int firstDirY, int firstPointX, int firstPointY, int vrtx,
    float invTimeHorizonObst, int* nAgents, float* x, float* y, float* radius, float* veloX,
    float* veloY, float* vrtxPointX, float* vrtxPointY, float* unitDirX, float* unitDirY,
    int* prevVrtcs, int* nextVrtcs, bool* isConvex, float* orcaLines, int* nLines)
{
    //vrtx = obstacle point 1
    int nextVrtx = nextVrtcs[vrtx];   //obstacle point 2

    float const relPosX1 = vrtxPointX[vrtx] - x[self];
    float const relPosY1 = vrtxPointY[vrtx] - y[self];
    float const relPosX2 = vrtxPointX[nextVrtx] - x[self];
    float const relPosY2 = vrtxPointY[nextVrtx] - y[self];

    /*  * Check if velocity obstacle of obstacle is already taken care of by
        * previously constructed obstacle ORCA lines.
        */
    for (int j = 0; j < nLines[self]; ++j)
        if (det(invTimeHorizonObst * relPosX1 - orcaLines[firstPointX + j],
            invTimeHorizonObst * relPosY1 - orcaLines[firstPointY + j],
            orcaLines[firstDirX + j], orcaLines[firstDirY + j]) - 
            invTimeHorizonObst * radius[self] >= -RVO_EPSILON &&
            det(invTimeHorizonObst * relPosX2 - orcaLines[firstPointX + j],
                invTimeHorizonObst * relPosY2 - orcaLines[firstPointY + j],
                orcaLines[firstDirX + j], orcaLines[firstDirY + j]) - 
            invTimeHorizonObst * radius[self] >= -RVO_EPSILON)
            return false;

    /* Not yet covered. Check for collisions. */
    float const distSq1 = absSq(relPosX1, relPosY1);
    float const distSq2 = absSq(relPosX2, relPosY2);

    float const radiusSq = sqr(radius[self]);

    float const obstVectX = vrtxPointX[nextVrtx] - vrtxPointX[vrtx];
    float const obstVectY = vrtxPointY[nextVrtx] - vrtxPointY[vrtx];

    float const s = prod(-relPosX1, -relPosY1, obstVectX, obstVectY) /
        absSq(obstVectX, obstVectY);
    float const distSqLine = absSq(-relPosX1 - s * obstVectX, -relPosY1 - s *
        obstVectY);

    if (s < 0.0f && distSq1 <= radiusSq)
    {
        /* Collision with left vertex. Ignore if non-convex. */
        if (isConvex[vrtx])
        {
            orcaLines[firstPointX + nLines[self]] = 0.0f;
            orcaLines[firstPointY + nLines[self]] = 0.0f;
            orcaLines[firstDirX + nLines[self]] = norm(-relPosY1, relPosX1);
            orcaLines[firstDirY + nLines[self]] = norm(relPosX1, -relPosY1);
            return true;
        }
        return false;
    }
    else if (s > 1.0f && distSq2 <= radiusSq)
    {
        /*  * Collision with right vertex. Ignore if non-convex
            * or if it will be taken care of by neighoring obstace */
        if (isConvex[nextVrtx] && det(relPosX2, relPosY2, unitDirX[nextVrtx],
            unitDirY[nextVrtx]) >= 0.0f)
        {
            orcaLines[firstPointX + nLines[self]] = 0.0f;
            orcaLines[firstPointY + nLines[self]] = 0.0f;
            orcaLines[firstDirX + nLines[self]] = norm(-relPosY2, relPosX2);
            orcaLines[firstDirY + nLines[self]] = norm(relPosX2, -relPosY2);
            return true;
        }
        return false;
    }
    else if (s >= 0.0f && s < 1.0f && distSqLine <= radiusSq)
    {
        /* Collision with obstacle segment. */
        orcaLines[firstPointX + nLines[self]] = 0.0f;
        orcaLines[firstPointY + nLines[self]] = 0.0f;
        orcaLines[firstDirX + nLines[self]] = -unitDirX[vrtx];
        orcaLines[firstDirY + nLines[self]] = -unitDirY[vrtx];
        return true;
    }

    ////adding this fixes the isue where agents collide next to the left and top
    ////side of objects. Clearly something below is not right..
    //return false;

    /*  * No collision.
        * Compute legs. When obliquely viewed, both legs can come from a single
        * vertex. Legs extend cut-off line when nonconvex vertex.
        */
    float leftLegDirX, leftLegDirY, rightLegDirX, rightLegDirY;

    if (s < 0.0f && distSqLine <= radiusSq)
    {
        /*  * Obstacle viewed obliquely so that left vertex
            * defines velocity obstacle.
            */
        if (!isConvex[vrtx])
            return false;               //Ignore obstacle

        nextVrtx = vrtx;                //obstacle2 = obstacle1;        

        const float leg1 = sqrtf(distSq1 - radiusSq);
        leftLegDirX = (relPosX1 * leg1 - relPosY1 * radius[self]) / distSq1;
        leftLegDirY = (relPosX1 * radius[self] + relPosY1 * leg1) / distSq1;
        rightLegDirX = (relPosX1 * leg1 + relPosY1 * radius[self]) / distSq1;
        rightLegDirY = (-relPosX1 * radius[self] + relPosY1 * leg1) / distSq1;
    }
    else if (s > 1.0f && distSqLine <= radiusSq)
    {
        /*  * Obstacle viewed obliquely so that
            * right vertex defines velocity obstacle.
            */
        if (!isConvex[nextVrtx])
            return false;               //Ignore obstacle

        vrtx = nextVrtx;                //obstacle1 = obstacle2;        

        const float leg2 = sqrtf(distSq2 - radiusSq);
        leftLegDirX = (relPosX2 * leg2 - relPosY2 * radius[self]) / distSq2;
        leftLegDirY = (relPosX2 * radius[self] + relPosY2 * leg2) / distSq2;
        rightLegDirX = (relPosX2 * leg2 + relPosY2 * radius[self]) / distSq2;
        rightLegDirY = (-relPosX2 * radius[self] + relPosY2 * leg2) / distSq2;
    }
    else
    {
        if (isConvex[vrtx])             //Usual situation
        {
            const float leg1 = sqrtf(distSq1 - radiusSq);
            leftLegDirX = (relPosX1 * leg1 - relPosY1 * radius[self]) / distSq1;
            leftLegDirY = (relPosX1 * radius[self] + relPosY1 * leg1) / distSq1;
        }
        else                            //Left vertex non-convex; left leg extends cut-off line
        {
            leftLegDirX = -unitDirX[vrtx];
            leftLegDirY = -unitDirY[vrtx];
        }

        if (isConvex[nextVrtx])
        {
            const float leg2 = sqrtf(distSq2 - radiusSq);
            rightLegDirX = (relPosX2 * leg2 + relPosY2 * radius[self]) / distSq2;
            rightLegDirY = (-relPosX2 * radius[self] + relPosY2 * leg2) / distSq2;
        }
        else                            //Right vertex non-convex; right leg extends cut-off line
        {
            rightLegDirX = unitDirX[vrtx];
            rightLegDirY = unitDirY[vrtx];
        }
    }

    /*  * Legs can never point into neighboring edge when convex vertex,
        * take cutoff-line of neighboring edge instead. If velocity projected on
        * "foreign" leg, no constraint is added.
        */

    const int prevVrtx = prevVrtcs[vrtx];
    bool isLeftLegForeign = false;
    bool isRightLegForeign = false;

    if (isConvex[vrtx] && det(leftLegDirX, leftLegDirY, -unitDirX[prevVrtx],
        -unitDirY[prevVrtx]) >= 0.0f)
    {                                   //Left leg points into obstacle        
        leftLegDirX = -unitDirX[prevVrtx];
        leftLegDirY = -unitDirY[prevVrtx];
        isLeftLegForeign = true;
    }

    if (isConvex[nextVrtx] && det(rightLegDirX, rightLegDirY,
        unitDirX[nextVrtx], unitDirY[nextVrtx]) <= 0.0f)
    {                                   //Right leg points into obstacle        
        rightLegDirX = unitDirX[nextVrtx];
        rightLegDirY = unitDirY[nextVrtx];
        isRightLegForeign = true;
    }

    /* Compute cut-off centers. */
    const float leftCutOffX = invTimeHorizonObst * (vrtxPointX[vrtx] - x[self]);
    const float leftCutOffY = invTimeHorizonObst * (vrtxPointY[vrtx] - y[self]);
    const float rightCutOffX = invTimeHorizonObst * (vrtxPointX[nextVrtx] -
        x[self]);
    const float rightCutOffY = invTimeHorizonObst * (vrtxPointY[nextVrtx] -
        y[self]);
    const float cutOffVecX = rightCutOffX - leftCutOffX;
    const float cutOffVecY = rightCutOffY - leftCutOffY;

    /* Project current velocity on velocity obstacle. */

    /* Check if current velocity is projected on cutoff circles. */
    const float t = (vrtx == nextVrtx ? 0.5f : prod(veloX[self] - leftCutOffX,
        veloY[self] - leftCutOffY, cutOffVecX, cutOffVecY) / absSq(cutOffVecX,
            cutOffVecY));
    const float tLeft = prod(veloX[self] - leftCutOffX, veloY[self] -
        leftCutOffY, leftLegDirX, leftLegDirY);
    const float tRight = prod(veloX[self] - rightCutOffX, veloY[self] -
        rightCutOffY, rightLegDirX, rightLegDirY);

    if ((t < 0.0f && tLeft < 0.0f) || (vrtx == nextVrtx && tLeft < 0.0f &&
        tRight < 0.0f))
    {
        /* Project on left cut-off circle. */
        const float unitWX = norm(veloX[self] - leftCutOffX, veloY[self] -
            leftCutOffY);
        const float unitWY = norm(veloY[self] - leftCutOffY, veloX[self] -
            leftCutOffX);

        orcaLines[firstDirX + nLines[self]] = unitWY;
        orcaLines[firstDirY + nLines[self]] = -unitWX;
        orcaLines[firstPointX + nLines[self]] = leftCutOffX + radius[self] *
            invTimeHorizonObst * unitWX;
        orcaLines[firstPointY + nLines[self]] = leftCutOffY + radius[self] *
            invTimeHorizonObst * unitWY;
        return true;
    }
    else if (t > 1.0f && tRight < 0.0f)
    {
        /* Project on right cut-off circle. */
        const float unitWX = norm(veloX[self] - rightCutOffX, veloY[self] -
            rightCutOffY);
        const float unitWY = norm(veloY[self] - rightCutOffY, veloX[self] -
            rightCutOffX);

        orcaLines[firstDirX + nLines[self]] = unitWY;
        orcaLines[firstDirY + nLines[self]] = -unitWX;
        orcaLines[firstPointX + nLines[self]] = rightCutOffX + radius[self] *
            invTimeHorizonObst * unitWX;
        orcaLines[firstPointY + nLines[self]] = rightCutOffY + radius[self] *
            invTimeHorizonObst * unitWY;
        return true;
    }

    /*  * Project on left leg, right leg, or cut-off line, whichever is closest
        * to velocity.
        */
    const float distSqCutoff = ((t < 0.0f || t > 1.0f || vrtx == nextVrtx) ?
        inf() : absSq(veloX[self] - (leftCutOffX + t * cutOffVecX), 
            veloY[self] - (leftCutOffY + t * cutOffVecY)));

    const float distSqLeft = ((tLeft < 0.0f) ?
        inf() : absSq(veloX[self] - (leftCutOffX + tLeft * leftLegDirX), 
            veloY[self] - (leftCutOffY + tLeft * leftLegDirY)));

    const float distSqRight = ((tRight < 0.0f) ?
        inf() : absSq(veloX[self] - (rightCutOffX + tRight * rightLegDirX), 
            veloY[self] - (rightCutOffY + tRight * rightLegDirY)));

    if (distSqCutoff <= distSqLeft && distSqCutoff <= distSqRight)
    {
        /* Project on cut-off line. */
        orcaLines[firstDirX + nLines[self]] = -unitDirX[vrtx];
        orcaLines[firstDirY + nLines[self]] = -unitDirY[vrtx];
        orcaLines[firstPointX + nLines[self]] = leftCutOffX +
            radius[self] * invTimeHorizonObst * (-orcaLines[firstDirY +
                nLines[self]]);
        orcaLines[firstPointY + nLines[self]] = leftCutOffY +
            radius[self] * invTimeHorizonObst * orcaLines[firstDirX +
            nLines[self]];
        return true;
    }
    else if (distSqLeft <= distSqRight)
    {
        /* Project on left leg. */
        if (isLeftLegForeign) {
            return false;
        }

        orcaLines[firstDirX + nLines[self]] = leftLegDirX;
        orcaLines[firstDirY + nLines[self]] = leftLegDirY;
        orcaLines[firstPointX + nLines[self]] = leftCutOffX +
            radius[self] * invTimeHorizonObst * (-orcaLines[firstDirY +
                nLines[self]]);
        orcaLines[firstPointY + nLines[self]] = leftCutOffY +
            radius[self] * invTimeHorizonObst * orcaLines[firstDirX +
            nLines[self]];
        return true;
    }
    else
    {
        /* Project on right leg. */
        if (isRightLegForeign) {
            return false;
        }

        orcaLines[firstDirX + nLines[self]] = -rightLegDirX;
        orcaLines[firstDirY + nLines[self]] = -rightLegDirY;
        orcaLines[firstPointX + nLines[self]] = rightCutOffX +
            radius[self] * invTimeHorizonObst * (-orcaLines[firstDirY +
                nLines[self]]);
        orcaLines[firstPointY + nLines[self]] = rightCutOffY +
            radius[self] * invTimeHorizonObst * orcaLines[firstDirX +
            nLines[self]];
        return true;
    }

    return false;
}

__device__ void agentLine(
    int self, int firstDirX, int firstDirY, int firstPointX, int firstPointY, int other,
    float* timeStep, float* posX, float* posY, float* radius, float* veloX, float* veloY,
    float invTimeHorizonAgnt, float* orcaLines, int* nLines, float* prefPersDist)
{
    float const relPosX = posX[other] - posX[self];
    float const relPosY = posY[other] - posY[self];
    float const relVelX = veloX[self] - veloX[other];
    float const relVelY = veloY[self] - veloY[other];
    float const distSq = absSq(relPosX, relPosY);
    float combinedRadius = radius[self] + radius[other] + prefPersDist[self];;
    float const combinedRadiusSq = sqr(combinedRadius);

    float lineDirX;
    float lineDirY;
    float uX;
    float uY;

    if (distSq > combinedRadiusSq)      //no collision
    {
        /* w is the vector from cutoff center to relative velocity. */
        const float wX = relVelX - invTimeHorizonAgnt * relPosX;
        const float wY = relVelY - invTimeHorizonAgnt * relPosY;
        const float wLengthSq = absSq(wX, wY);
        const float dotProduct1 = prod(wX, wY, relPosX, relPosY);

        if (dotProduct1 < 0.0f && sqr(dotProduct1) > combinedRadiusSq *
            wLengthSq)
        {                               //Project on cut-off circle
            const float wLength = sqrtf(wLengthSq);
            const float unitWX = wX / wLength;
            const float unitWY = wY / wLength;

            lineDirX = unitWY;
            lineDirY = -unitWX;
            uX = (combinedRadius * invTimeHorizonAgnt - wLength) * unitWX;
            uY = (combinedRadius * invTimeHorizonAgnt - wLength) * unitWY;
        }
        else
        {                               //Project on legs
            const float leg = sqrtf(distSq - combinedRadiusSq);

            //determinant of the 2D square matrix
            if (det(relPosX, relPosY, wX, wY) > 0.0f)
            {                           //Project on left leg
                lineDirX = (relPosX * leg - relPosY * combinedRadius) / distSq;
                lineDirY = (relPosX * combinedRadius + relPosY * leg) / distSq;
            }
            else
            {                           //Project on right leg
                lineDirX = -(relPosX * leg + relPosY * combinedRadius) / distSq;
                lineDirY = -(-relPosX * combinedRadius + relPosY * leg) /
                    distSq;
            }
            const float dotProduct2 = prod(relVelX, relVelY, lineDirX,
                lineDirY);

            uX = dotProduct2 * lineDirX - relVelX;
            uY = dotProduct2 * lineDirY - relVelY;
        }
    }
    else                                //collision
    {
        /* Collision. Project on cut-off circle of time timeStep. */
        const float invTimeStep = 1.0f / timeStep[0];

        /* Vector from cutoff center to relative velocity. */
        const float wX = relVelX - invTimeStep * relPosX;
        const float wY = relVelY - invTimeStep * relPosY;

        const float wLength = abs(wX, wY);
        const float unitWX = wX / wLength;
        const float unitWY = wY / wLength;

        lineDirX = unitWY;
        lineDirY = -unitWX;

        uX = (combinedRadius * invTimeStep - wLength) * unitWX;
        uY = (combinedRadius * invTimeStep - wLength) * unitWY;
    }

    orcaLines[firstPointX + nLines[self]] = veloX[self] + 0.5f * uX;
    orcaLines[firstPointY + nLines[self]] = veloY[self] + 0.5f * uY;
    orcaLines[firstDirX + nLines[self]] = lineDirX;
    orcaLines[firstDirY + nLines[self]] = lineDirY;

    ++nLines[self];
}

__device__ bool linearProgram1(
    int self, int firstDirX, int firstDirY, int firstPointX, int firstPointY, int lineIdx,
    float radius, float prefVeloX, float prefVeloY, bool optimizeDirection, float* orcaLines,
    float* newVeloX, float* newVeloY)
{
    const float dotProduct = prod(orcaLines[firstPointX + lineIdx],
        orcaLines[firstPointY + lineIdx],
        orcaLines[firstDirX + lineIdx],
        orcaLines[firstDirY + lineIdx]);
    const float discriminant = sqr(dotProduct) + sqr(radius) - absSq(
        orcaLines[firstPointX + lineIdx],
        orcaLines[firstPointY + lineIdx]);

    if (discriminant < 0.0f)
        return false;                   //Max speed circle fully invalidates line lineNo

    const float sqrtDiscriminant = sqrtf(discriminant);
    float tLeft = -dotProduct - sqrtDiscriminant;
    float tRight = -dotProduct + sqrtDiscriminant;

    for (int idx = 0; idx < lineIdx; ++idx)
    {
        const float denominator = det(
            orcaLines[firstDirX + lineIdx],
            orcaLines[firstDirY + lineIdx],
            orcaLines[firstDirX + idx],
            orcaLines[firstDirY + idx]);
        const float numerator = det(
            orcaLines[firstDirX + idx],
            orcaLines[firstDirY + idx],
            orcaLines[firstPointX + lineIdx] - orcaLines[firstPointX + idx],
            orcaLines[firstPointY + lineIdx] - orcaLines[firstPointY + idx]);

        if (fabs(denominator) <= RVO_EPSILON)
        {
            if (numerator < 0.0f)       //Lines lineNo and i are (almost) parallel
                return false;
            else
                continue;
        }

        const float t = numerator / denominator;

        if (denominator >= 0.0f)        //Line i bounds line lineNo on the right
            tRight = fmin(tRight, t);
        else                            //Line i bounds line lineNo on the left
            tLeft = fmax(tLeft, t);

        if (tLeft > tRight)
            return false;
    }

    if (optimizeDirection)              //Optimize direction
    {                                   //Take right extreme
        if (prod(prefVeloX, prefVeloY,
            orcaLines[firstDirX + lineIdx],
            orcaLines[firstDirY + lineIdx]) > 0.0f)
        {
            newVeloX[self] = orcaLines[firstPointX + lineIdx] + tRight *
                orcaLines[firstDirX + lineIdx];
            newVeloY[self] = orcaLines[firstPointY + lineIdx] + tRight *
                orcaLines[firstDirY + lineIdx];
        }
        else                            //Take left extreme
        {
            newVeloX[self] = orcaLines[firstPointX + lineIdx] + tLeft *
                orcaLines[firstDirX + lineIdx];
            newVeloY[self] = orcaLines[firstPointY + lineIdx] + tLeft *
                orcaLines[firstDirY + lineIdx];
        }
    }
    else                                //Optimize closest point
    {
        const float t = prod(orcaLines[firstDirX + lineIdx],
            orcaLines[firstDirY + lineIdx],
            prefVeloX - orcaLines[firstPointX + lineIdx],
            prefVeloY - orcaLines[firstPointY + lineIdx]);

        if (t < tLeft)
        {
            newVeloX[self] = orcaLines[firstPointX + lineIdx] + tLeft *
                orcaLines[firstDirX + lineIdx];
            newVeloY[self] = orcaLines[firstPointY + lineIdx] + tLeft *
                orcaLines[firstDirY + lineIdx];
        }
        else if (t > tRight)
        {
            newVeloX[self] = orcaLines[firstPointX + lineIdx] + tRight *
                orcaLines[firstDirX + lineIdx];
            newVeloY[self] = orcaLines[firstPointY + lineIdx] + tRight *
                orcaLines[firstDirY + lineIdx];
        }
        else
        {
            newVeloX[self] = orcaLines[firstPointX + lineIdx] + t *
                orcaLines[firstDirX + lineIdx];
            newVeloY[self] = orcaLines[firstPointY + lineIdx] + t *
                orcaLines[firstDirY + lineIdx];
        }
    }
    return true;
}

__device__ int linearProgram2(
    int self, int firstDirX, int firstDirY, int firstPointX, int firstPointY, float radius,
    float prefVeloX, float prefVeloY, bool optimizeDirection, float* orcaLines, int nLines,
    float* newVeloX, float* newVeloY)
{
    if (optimizeDirection)              //Optimize direction
    {
        newVeloX[self] = prefVeloX * radius;
        newVeloY[self] = prefVeloY * radius;
    }
    //Optimize closest point and outside circle
    else if (absSq(prefVeloX, prefVeloY) > sqr(radius))
    {
        newVeloX[self] = norm(prefVeloX, prefVeloY) * radius;
        newVeloY[self] = norm(prefVeloY, prefVeloX) * radius;
    }
    else                                //Optimize closest point and inside circle
    {
        newVeloX[self] = prefVeloX;
        newVeloY[self] = prefVeloY;
    }

    for (int idx = 0; idx < nLines; ++idx)
    {
        if (det(orcaLines[firstDirX + idx],
            orcaLines[firstDirY + idx],
            orcaLines[firstPointX + idx] - newVeloX[self],
            orcaLines[firstPointY + idx] - newVeloY[self]) > 0.0f)
        {
            float const tempResultX = newVeloX[self];
            float const tempResultY = newVeloY[self];

            if (!linearProgram1(self, firstDirX, firstDirY, firstPointX,
                firstPointY, idx, radius, prefVeloX, prefVeloY,
                optimizeDirection, orcaLines, newVeloX, newVeloY))
            {
                newVeloX[self] = tempResultX;
                newVeloY[self] = tempResultY;
                return idx;
            }
        }
    }
    return nLines;
}

__device__ void linearProgram3(
    int self, int firstDirX, int firstDirY, int firstPointX, int firstPointY, int numObstLines,  
    int beginLine, float radius, float* orcaLines, float* projLines, int nLines, 
    float* newVeloX, float* newVeloY)
{
    float distance = 0.0f;

    for (int idx = beginLine; idx < nLines; ++idx)
    {

        if (det(orcaLines[firstDirX + idx],
            orcaLines[firstDirY + idx],
            orcaLines[firstPointX + idx] - newVeloX[self],
            orcaLines[firstPointY + idx] - newVeloY[self]) > distance)
        {
            /* Result does not satisfy constraint of line idx */
            int numProjLines = numObstLines;

            //for all neighLines before beginLine (not obst)
            for (int j = numObstLines; j < idx; ++j) 
            {
                float determinant = det(orcaLines[firstDirX + idx],
                    orcaLines[firstDirY + idx],
                    orcaLines[firstDirX + j],
                    orcaLines[firstDirY + j]);

                if (fabs(determinant) <= RVO_EPSILON)
                {                        //Line idx and line j are parallel                                       
                    if (prod(orcaLines[firstDirX + idx],
                        orcaLines[firstDirY + idx],
                        orcaLines[firstDirX + j],
                        orcaLines[firstDirY + j]) > 0.0f)
                        continue;       //Line idx and line j point in the same direction
                    else
                    {                   //Line idx and line j point in opposite direction
                        projLines[firstPointX + numProjLines] = 0.5f * (
                            orcaLines[firstPointX + idx] +
                            orcaLines[firstPointX + j]);
                        projLines[firstPointY + numProjLines] = 0.5f * (
                            orcaLines[firstPointY + idx] +
                            orcaLines[firstPointY + j]);
                    }
                }
                else
                {
                    const float newDet = det(orcaLines[firstDirX + j],
                        orcaLines[firstDirY + j],
                        orcaLines[firstPointX + idx] - orcaLines[firstPointX +
                        j],
                        orcaLines[firstPointY + idx] - orcaLines[firstPointY +
                        j]);

                    projLines[firstPointX + numProjLines] =
                        orcaLines[firstPointX + idx] + (newDet / determinant) *
                        orcaLines[firstDirX + idx];
                    projLines[firstPointY + numProjLines] =
                        orcaLines[firstPointY + idx] + (newDet / determinant) *
                        orcaLines[firstDirY + idx];
                }

                projLines[firstDirX + numProjLines] = norm(
                    orcaLines[firstDirX + j] - orcaLines[firstDirX + idx],
                    orcaLines[firstDirY + j] - orcaLines[firstDirY + idx]);
                projLines[firstDirY + numProjLines] = norm(
                    orcaLines[firstDirY + j] - orcaLines[firstDirY + idx],
                    orcaLines[firstDirX + j] - orcaLines[firstDirX + idx]);

                ++numProjLines;
            }

            float const tempResultX = newVeloX[self];
            float const tempResultY = newVeloY[self];

            if (linearProgram2(self, firstDirX, firstDirY, firstPointX,
                firstPointY, radius, -orcaLines[firstDirY + idx],
                orcaLines[firstDirX + idx], true, projLines, numProjLines,
                newVeloX, newVeloY) < numProjLines)
            {
                /* This should in principle not happen.  The result is by definition
                 * already in the feasible region of this linear program. If it fails,
                 * it is due to small floating point error, and the current result is
                 * kept.
                 */
                newVeloX[self] = tempResultX;
                newVeloY[self] = tempResultY;
            }

            distance = det(orcaLines[firstDirX + idx],
                orcaLines[firstDirY + idx],
                orcaLines[firstPointX + idx] - newVeloX[self],
                orcaLines[firstPointY + idx] - newVeloY[self]);
        }
    }
}