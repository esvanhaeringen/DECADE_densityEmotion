#include "BaseModel.h"

void BaseModel::update()
{
	d_step += 1;
	updatePerception();
	updateEmotion();
	updateBehaviour();
	if (d_outputPath != "")
		writeStep();
}

