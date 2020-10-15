#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QKeyEvent>
#include <QDebug>
#include "mainwindow.h"

/*      Default constuctor function
*/
MainWindow::MainWindow(QWidget *parent):
    QMainWindow(parent)
    , fileMenu(parent)
    , viewMenu(nullptr)
//    , currentImage(nullptr)
{
    initUI();
}

MainWindow::~MainWindow(){
}

/*      This function design initUI()
*/
void MainWindow::initUI(){
    this->resize(800, 600);
    //      setup menubar
    fileMenu = menuBar()->addMenu("&File");
    viewMenu = menuBar()->addMenu("&View");

    //      set toolbar
    fileToolBar = addToolBar("File");
    viewToolBar = addToolBar("View");

    //      main area for image display
    imageScence = new QGraphicsScene(this);
    imageView = new QGraphicsView(imageScence);
    setCentralWidget(imageView);

    //      Setup status bar
    mainStatusBar = statusBar();
    mainStatusLabel = new QLabel(mainStatusBar);
    mainStatusBar->addPermanentWidget(mainStatusLabel);
    mainStatusLabel->setText("Image information will be here !");
}
