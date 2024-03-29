/*
 * The MIT License
 *
 * Copyright 2021 Marcelo.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package org.tensa.facecheck.buffer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Buffer de datos que provee inputStream y outputStream para transferencia de
 * datos con economia de espacio con uso de subplanos, pensado para
 * convoluciones, este buffer funciona independiennte de datos float o double
 *
 * @see InputStream
 * @see OutputStream
 * @see Double
 * @see Float
 * @author Marcelo
 */
public class PlaneBufferedData {

    private final int offx;
    private final int offy;
    private final int width;
    private final int height;
    private final int windWidth;
    private final int windHeight;
    private final int typeSize;
    private final int bands;
    private final int samples;
    private final byte[] data;

    /**
     * crea un nuevo plano limitado por sus parametros, asume la cantidad de
     * samples y bandas en 1
     *
     * @param width ancho del plano
     * @param height alto del plano
     * @param typeSize tamaño del dato a guardar
     */
    public PlaneBufferedData(int width, int height, int typeSize) {
        this(width, height, typeSize, 1);
    }

    /**
     * crea un nuevo plano limitado por sus parametros, asume la cantidad de
     * samples 1
     *
     * @param width ancho del plano
     * @param height alto del plano
     * @param typeSize tamaño del dato a guardar
     * @param bands nuero de bandas o canales
     */
    public PlaneBufferedData(int width, int height, int typeSize, int bands) {
        this(width, height, typeSize, bands, 1);
    }

    /**
     * crea un nuevo plano limitado por sus parametros
     *
     * @param width ancho del plano
     * @param height alto del plano
     * @param typeSize tamaño del dato a guardar
     * @param bands nuero de bandas o canales
     * @param samples numero de muestras totales
     */
    public PlaneBufferedData(int width, int height, int typeSize, int bands, int samples) {
        this(0, 0, width, height, width, height, typeSize, bands, samples, new byte[width * height * typeSize * bands * samples]);
    }

    /**
     * crea un nuevo plano limitado por sus parametros
     *
     * @param offx limite superior izquierdo de la ventana de datos a acceder
     * @param offy limite superior izquierdo de la ventana de datos a acceder
     * @param width ancho del plano
     * @param height alto del plano
     * @param windWidth ancho y alto limite de la ventana de datos a acceder
     * @param windHeight ancho y alto limite de la ventana de datos a acceder
     * @param typeSize tamaño del dato a guardar
     * @param bands nuero de bandas o canales
     * @param samples numero de muestras totales
     * @param data datos originales que seran accedidos con los nuevos limites
     * de la ventana definida
     */
    public PlaneBufferedData(int offx, int offy, int width, int height, int windWidth, int windHeight, int typeSize, int bands, int samples, byte[] data) {
        this.offx = offx;
        this.offy = offy;
        this.width = width;
        this.height = height;
        this.windWidth = windWidth;
        this.windHeight = windHeight;
        this.typeSize = typeSize;
        this.bands = bands;
        this.samples = samples;
        this.data = data;
    }

    /**
     * produce un PlaneBufferedData basado en un subconcunto de datos del
     * PlaneBufferedData actual, limitado por la ventana indicada en sus
     * parametros
     *
     * @param x limite superior izquierdo de la ventana
     * @param y limite superior izquierdo de la ventana
     * @param w ancho de la ventana
     * @param h alto de la ventana
     * @return PlaneBufferedData (sub plano)
     */
    public PlaneBufferedData getSubPlaneBufferedData(int x, int y, int w, int h) {
        if (x < 0) {
            throw new IllegalArgumentException("x is negative");
        }

        if (y < 0) {
            throw new IllegalArgumentException("y is negative");
        }

        if (w < 0) {
            throw new IllegalArgumentException("width is negative");
        }

        if (h < 0) {
            throw new IllegalArgumentException("height is negative");
        }

        if (windWidth < w) {
            throw new IllegalArgumentException("width too big");
        }

        if (windHeight < h) {
            throw new IllegalArgumentException("height too big");
        }

        if (windWidth < x) {
            throw new IllegalArgumentException("x too big");
        }

        if (windHeight < y) {
            throw new IllegalArgumentException("y too big");
        }

        return new PlaneBufferedData(x + offx, y + offy, width, height, w, h, typeSize, bands, samples, data);
    }

    /**
     * provee OutputStream que apunta el repositorio interno del
     * PlaneBufferedData
     *
     * @return OutputStream
     */
    public OutputStream getOutputStreamBuffer() {
        return new BufferedOutputStream(new OutputStream() {
            private int count = 0;

            @Override
            public void write(int b) throws IOException {
                byte myByte = (byte) b;
                int indice = this.count++ / typeSize;
                int band = indice % bands;
                int pixel = indice / bands;
                int x = pixel % windWidth;
                int y = pixel / windWidth;
                int x2 = x + offx;
                int y2 = y % windHeight + offy;
                int sample = y / windHeight;

                int ncount = sample * width * height * bands * typeSize
                        + y2 * width * bands * typeSize
                        + x2 * bands * typeSize
                        + band * typeSize;

                data[ncount] = myByte;
            }

//        }, windWidth * windHeight * typeSize * bands * samples);
        }, windWidth * windHeight * bands * typeSize);
    }

    /**
     * provee InputStream que apunta al repositorio interno del
     * PlaneBufferedData
     *
     * @return InputStream
     */
    public InputStream getInputStreamBuffer() {
        return new BufferedInputStream(new InputStream() {
            private int count = 0;

            @Override
            public int read() throws IOException {
                int indice = this.count++ / typeSize;
                int band = indice % bands;
                int pixel = indice / bands;
                int x = pixel % windWidth;
                int y = pixel / windWidth;
                int x2 = x + offx;
                int y2 = y % windHeight + offy;
                int sample = y / windHeight;

                int ncount = sample * width * height * bands * typeSize
                        + y2 * width * bands * typeSize
                        + x2 * bands * typeSize
                        + band * typeSize;

                return data[ncount] & 0xff;
            }
//        }, windWidth * windHeight * typeSize * bands * samples);
        }, windWidth * windHeight * bands * typeSize);
    }

}
