QT += core gui widgets

TARGET = blur
TEMPLATE = app

QMAKE_CXXFLAGS += -fopenmp

LIBS += -L../fftw3 -lfftw3f-3 -fopenmp
INCLUDEPATH += ../fftw3

SOURCES += blur.cpp
