QT += core gui widgets

TARGET = blur
TEMPLATE = app

QMAKE_CXXFLAGS += -fopenmp

win32 {
    LIBS += -L../fftw3 -lfftw3f-3 -fopenmp
    INCLUDEPATH += ../fftw3
}

linux {
    LIBS += -lfftw3f -lfftw3f_omp -lgomp
}

SOURCES += blur.cpp
