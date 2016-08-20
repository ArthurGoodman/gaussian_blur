#include <QApplication>

#include <omp.h>
#include <fftw3.h>
#include <functional>
#include <QImage>
#include <QLabel>

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

    fftwf_complex *input, *filter, *output;
    fftwf_plan forward_plan, backward_plan;

    QImage image("image.jpg"), blurred(image.width(), image.height(), QImage::Format_RGB32);
    int width = image.width(), height = image.height();

    std::function<int(int, int)> index = [&](int x, int y) -> int {
        return (x + width) % width + (y + height) % height * width;
    };

    fftwf_init_threads();

    fftwf_plan_with_nthreads(omp_get_max_threads());

    input = (fftwf_complex *)fftwf_malloc(width * height * sizeof(fftwf_complex));
    filter = (fftwf_complex *)fftwf_malloc(width * height * sizeof(fftwf_complex));
    output = (fftwf_complex *)fftwf_malloc(width * height * sizeof(fftwf_complex));

    forward_plan = fftwf_plan_dft_2d(height, width, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    backward_plan = fftwf_plan_dft_2d(height, width, output, output, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftwf_plan filter_plan = fftwf_plan_dft_2d(height, width, filter, filter, FFTW_FORWARD, FFTW_ESTIMATE);

    std::fill((float *)filter, (float *)(filter + width * height), 0.0f);

    filter[index(-2, -2)][0] = 0.003765;
    filter[index(-1, -2)][0] = 0.015019;
    filter[index(0, -2)][0] = 0.023792;
    filter[index(1, -2)][0] = 0.015019;
    filter[index(2, -2)][0] = 0.003765;

    filter[index(-2, -1)][0] = 0.015019;
    filter[index(-1, -1)][0] = 0.059912;
    filter[index(0, -1)][0] = 0.094907;
    filter[index(1, -1)][0] = 0.059912;
    filter[index(2, -1)][0] = 0.015019;

    filter[index(-2, 0)][0] = 0.023792;
    filter[index(-1, 0)][0] = 0.094907;
    filter[index(0, 0)][0] = 0.150342;
    filter[index(1, 0)][0] = 0.094907;
    filter[index(2, 0)][0] = 0.023792;

    filter[index(-2, 1)][0] = 0.015019;
    filter[index(-1, 1)][0] = 0.059912;
    filter[index(0, 1)][0] = 0.094907;
    filter[index(1, 1)][0] = 0.059912;
    filter[index(2, 1)][0] = 0.015019;

    filter[index(-2, 2)][0] = 0.003765;
    filter[index(-1, 2)][0] = 0.015019;
    filter[index(0, 2)][0] = 0.023792;
    filter[index(1, 2)][0] = 0.015019;
    filter[index(2, 2)][0] = 0.003765;

    fftwf_execute(filter_plan);
    fftwf_destroy_plan(filter_plan);

    for (int c = 0; c < 3; c++) {
        std::fill((float *)input, (float *)(input + width * height), 0.0f);

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                input[index(x, y)][0] = getChannel(image.pixel(x, y), c);

        fftwf_execute(forward_plan);

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                output[index(x, y)][0] *= filter[index(x, y)][0];
                output[index(x, y)][1] *= filter[index(x, y)][0];
            }

        fftwf_execute(backward_plan);

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                blurred.setPixel(x, y, setChannel(blurred.pixel(x, y), c, round(output[index(x, y)][0]) / width / height));
    }

    fftwf_free(input);
    fftwf_free(filter);
    fftwf_free(output);

    fftwf_destroy_plan(forward_plan);
    fftwf_destroy_plan(backward_plan);

    fftwf_cleanup_threads();

    QLabel label;
    label.setPixmap(QPixmap::fromImage(blurred));
    label.show();

    blurred.save("blurred.bmp");

    return app.exec();
}
