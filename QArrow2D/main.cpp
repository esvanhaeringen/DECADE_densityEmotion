#include <iostream>
#include <windows.h>
#include "QArrow2D.h"
#include <QtWidgets/QApplication>

int main(int argc, char* argv[])
{
    QCoreApplication::setApplicationName("QArrow2D");
    QCoreApplication::setOrganizationName("Erik van Haeringen");
    QGuiApplication::setApplicationDisplayName(QCoreApplication::applicationName());
    QCoreApplication::setApplicationVersion(QT_VERSION_STR);
    QCoreApplication::setAttribute(Qt::AA_X11InitThreads);
    QCoreApplication::setAttribute(Qt::AA_DisableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_DisableWindowContextHelpButton);
    if (qgetenv("QT_FONT_DPI").isEmpty()) {
        qputenv("QT_FONT_DPI", "96");
    }
    QApplication app(argc, argv);
    QArrow2D mainWindow;
    mainWindow.show();

    return app.exec();
}