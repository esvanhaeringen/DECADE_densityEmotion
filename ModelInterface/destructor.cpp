#include "ModelInterface.h"

ModelInterface::~ModelInterface()
{
    if (d_modelType == 1)
        delete(d_decModel);
    else if (d_modelType == 2)
        delete(d_durModel);
}