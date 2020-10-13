#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QKeyEvent>
#include <QDebug>
#include "mainwindow.h"


void MainWindow::initUI(){
    this->resize(800, 600);
    //      setup menubar
    fileMenu = menuBar()->addMenu("&File");
    viewMenu = menuBar()->addMenu("&View");

    //      set toolbar
    fileToolBar = addToolBar("File");
    viewToolBar = addToolBar("View");

    //      main area for image display
    imageScene = new QGraphicsScene(this);

}
