#include <QApplication>

#include <fftw3.h>
#include <functional>
#include <omp.h>

#include <QtWidgets>

class Widget : public QWidget {
    QVBoxLayout vBox;
    QHBoxLayout hBox;
    QLabel label;
    QSlider slider;

    QImage &image;
    std::function<void(int)> compute;

public:
    Widget(QImage &image, const std::function<void(int)> &compute)
        : image(image)
        , compute(compute) {
        setFont(QFont("Courier New", 12));

        slider.setOrientation(Qt::Horizontal);
        slider.setMinimum(1);
        slider.setMaximum(100);
        slider.setSingleStep(1);
        slider.setPageStep(10);
        slider.setTickInterval(10);
        slider.setTickPosition(QSlider::TicksBelow);

        connect(&slider, &QSlider::valueChanged, this, &Widget::sliderValueChanged);

        vBox.addWidget(&label);

        hBox.addWidget(new QLabel("1", this));
        hBox.addWidget(&slider);
        hBox.addWidget(new QLabel("100", this));

        vBox.addLayout(&hBox);

        setLayout(&vBox);

        vBox.setSizeConstraint(QLayout::SetFixedSize);

        blur(1);
    }

protected:
    void keyPressEvent(QKeyEvent *e) {
        switch (e->key()) {
        case Qt::Key_Escape:
            close();
            break;

        case Qt::Key_Return:
            image.save("blurred.png");
            break;
        }
    }

private slots:
    void sliderValueChanged(int value) {
        blur(value);
    }

private:
    void blur(int sigma) {
        compute(sigma);
        label.setPixmap(QPixmap::fromImage(image));
    }
};

int getChannel(QRgb rgb, int c) {
    switch (c) {
    case 0:
        return qRed(rgb);
    case 1:
        return qGreen(rgb);
    case 2:
        return qBlue(rgb);
    default:
        return 0;
    }
}

QRgb setChannel(QRgb rgb, int c, int v) {
    switch (c) {
    case 0:
        return qRgb(v, qGreen(rgb), qBlue(rgb));
    case 1:
        return qRgb(qRed(rgb), v, qBlue(rgb));
    case 2:
        return qRgb(qRed(rgb), qGreen(rgb), v);
    default:
        return 0;
    }
}

int main(int argc, char **argv) {
    QApplication app(argc, argv);

    const int borderSize = 100;

    fftwf_complex *input, *filter, *output;
    fftwf_plan forward_plan, backward_plan;

    QImage image("image.jpg"), blurred(image.width(), image.height(), QImage::Format_RGB32);
    int width = image.width() + 2 * borderSize, height = image.height() + 2 * borderSize;

    auto index = [&](int x, int y) -> int {
        return (x + width) % width + (y + height) % height * width;
    };

    fftwf_init_threads();

    fftwf_plan_with_nthreads(omp_get_max_threads());

    input = (fftwf_complex *) fftwf_malloc(width * height * sizeof(fftwf_complex));
    filter = (fftwf_complex *) fftwf_malloc(width * height * sizeof(fftwf_complex));
    output = (fftwf_complex *) fftwf_malloc(width * height * sizeof(fftwf_complex));

    forward_plan = fftwf_plan_dft_2d(height, width, input, output, FFTW_FORWARD, FFTW_MEASURE);
    backward_plan = fftwf_plan_dft_2d(height, width, output, output, FFTW_BACKWARD, FFTW_MEASURE);

    fftwf_plan filter_plan = fftwf_plan_dft_2d(height, width, filter, filter, FFTW_FORWARD, FFTW_MEASURE);

    auto compute = [&](int sigma) {
        std::fill((float *) filter, (float *) (filter + width * height), 0.0f);

        const float a = 1.0 / 2 / M_PI / sigma / sigma;

        for (float x = -width / 2; x < width / 2; x++)
            for (float y = -height / 2; y < height / 2; y++)
                filter[index(x, y)][0] = a * expf(-(x * x + y * y) / 2 / sigma / sigma);

        fftwf_execute(filter_plan);

        for (int c = 0; c < 3; c++) {
            std::fill((float *) input, (float *) (input + width * height), 0.0f);

            // for (int x = 0; x < width - 2 * borderSize; x++)
            //     for (int y = 0; y < height - 2 * borderSize; y++)
            //         input[index(x + borderSize, y + borderSize)][0] = getChannel(image.pixel(x, y), c);

            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++) {
                    int nx = x - borderSize, ny = y - borderSize;

                    if (x < borderSize)
                        nx = abs(x - borderSize);

                    if (y < borderSize)
                        ny = abs(y - borderSize);

                    if (x > width - 1 - borderSize)
                        nx = 2 * width - 3 * borderSize - 1 - x;

                    if (y > height - 1 - borderSize)
                        ny = 2 * height - 3 * borderSize - 1 - y;

                    input[index(x, y)][0] = getChannel(image.pixel(nx, ny), c);
                }

            fftwf_execute(forward_plan);

            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++) {
                    output[index(x, y)][0] = output[index(x, y)][0] * filter[index(x, y)][0] - output[index(x, y)][1] * filter[index(x, y)][1];
                    output[index(x, y)][1] = output[index(x, y)][0] * filter[index(x, y)][1] + output[index(x, y)][1] * filter[index(x, y)][0];
                }

            fftwf_execute(backward_plan);

            for (int x = 0; x < width - 2 * borderSize; x++)
                for (int y = 0; y < height - 2 * borderSize; y++)
                    blurred.setPixel(x, y, setChannel(blurred.pixel(x, y), c, round(output[index(x + borderSize, y + borderSize)][0]) / width / height));
        }
    };

    //    fftwf_free(input);
    //    fftwf_free(filter);
    //    fftwf_free(output);

    //    fftwf_destroy_plan(forward_plan);
    //    fftwf_destroy_plan(backward_plan);

    //    fftwf_cleanup_threads();

    //    QLabel label;
    //    label.setPixmap(QPixmap::fromImage(blurred));
    //    label.show();

    //    blurred.save("blurred.bmp");

    Widget w(blurred, compute);
    w.show();

    return app.exec();
}
