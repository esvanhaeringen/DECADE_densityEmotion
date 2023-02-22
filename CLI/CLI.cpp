#include <iostream>
#include "../ModelInterface/ModelInterface.h"

using namespace std;

int main(int argc, char** argv)
{
    string d_templatePath;
    bool d_silent = false;
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp(argv[i], "-t") == 0 && (i + 1) < argc)
        {
            d_templatePath = string(argv[i + 1]);
            ++i;
        }
        if (strcmp(argv[i], "-silent") == 0)
            d_silent = true;
    }
    if (!d_silent)
        cout << "DECADE GPU [COMMAND-LINE INTERFACE]\n";
    if (d_templatePath == "")
    {
        if (!d_silent)
            cout << "Please add -t [path to config file] as an argument to run the model\n";
        return 0;
    }
    if (!d_silent)
        cout << "Setting up model with config file: " << d_templatePath << '\n';
    ModelInterface d_model(d_templatePath);
    float progress = 0.0;
    while (progress < 1.0) {
        d_model.update();
        progress = float(d_model.currentStep()) / d_model.endStep();
        if (!d_silent)
        {
            int barWidth = 70;
            std::cout << "Step " << d_model.currentStep() << '/' << d_model.endStep() << " [";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            std::cout.flush();
        }
    }
    if (!d_silent)
        std::cout << std::endl;
    return 0;
}