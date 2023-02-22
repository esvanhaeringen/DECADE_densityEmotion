#include "ConfigBuilder.h"
#include <iostream>
#include <windows.h>
#include <QtWidgets/QApplication>

int main(int argc, char* argv[])
{
    //AllocConsole();
    //freopen("CONOUT$", "w", stdout);
    //freopen("CONOUT$", "w", stderr);

    QCoreApplication::setApplicationName("Config Builder");
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
    ConfigBuilder mainWindow;
    mainWindow.show();

    return app.exec();
}