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
package org.tensa.facecheck.element.impl;

import org.tensa.facecheck.element.ElementData;
import org.tensa.facecheck.element.PanelElementData;

/**
 * Pannel de DataElements basados en arreglos float para multi proposito de
 * transporte reindexacion y acceso a datos
 *
 * @see PanelElementData
 * @see ElementData
 * @author Marcelo
 */
public class FloatPanelElementData implements PanelElementData {

    private final int width;
    private final int height;
    private final int offsetX;
    private final int offsetY;
    private final int windowWidth;
    private final int windowHeight;
    private final int channels;
    private final int samples;
    private final float[][][][] buffer;

    /**
     * creacion de un Panel Double en base a un arreglo y sus parametros, este
     * constructor se emplea de preferencia para subpannels
     *
     * @param width ancho del panel
     * @param height alto del panel
     * @param offsetX posicion de referencia de un subpannel
     * @param offsetY posicion de referencia de un subpannel
     * @param windowWidth ancho de ventana de referecia de un subpannel
     * @param windowHeight alto de ventana de referecia de un subpannel
     * @param channels cantidad de canales
     * @param samples cantiadad de muestras
     * @param buffer el arreglo original samples x channels x width x height
     */
    public FloatPanelElementData(int width, int height, int offsetX, int offsetY, int windowWidth, int windowHeight, int channels, int samples, float[][][][] buffer) {
        if (!(width > windowWidth || height > windowHeight
                || 0 <= offsetX || 0 <= offsetY)) {
            throw new IllegalArgumentException("argumento incorrecto");
        }

        this.width = width;
        this.height = height;
        this.offsetX = offsetX;
        this.offsetY = offsetY;
        this.windowWidth = windowWidth;
        this.windowHeight = windowHeight;
        this.channels = channels;
        this.samples = samples;
        this.buffer = buffer;
    }

    /**
     * creacion de un Panel Double en base a sus parametros, este creara el
     * arreglo interno de datos que se compartira con los subpanels crerados a
     * partir de este
     *
     * @param width ancho del panel
     * @param height alto del panel
     * @param channels cantidad de canales
     * @param samples cantiadad de muestras
     */
    public FloatPanelElementData(int width, int height, int channels, int samples) {
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.samples = samples;
        this.offsetX = 0;
        this.offsetY = 0;
        this.windowWidth = width;
        this.windowHeight = height;
        this.buffer = new float[samples][channels][width][height];
    }

    @Override
    public PanelElementData getSubPanel(int x, int y, int w, int h) {
        return new FloatPanelElementData(width, height, offsetX + x, offsetY + y, w, h, channels, samples, buffer);
    }

    @Override
    public void setSubPanel(int x, int y, int w, int h, PanelElementData panel) {
        FloatPanelElementData tmp = (FloatPanelElementData) panel;
        int minWidth = Integer.min(w, Integer.min(this.windowWidth - x, tmp.windowWidth));
        int minHeight = Integer.min(h, Integer.min(this.windowHeight - y, tmp.windowHeight));

        for (int esample = 0; esample < samples; esample++) {
            for (int echannel = 0; echannel < channels; echannel++) {
                for (int x1 = 0; x1 < minWidth; x1++) {
                    for (int y1 = 0; y1 < minHeight; y1++) {
                        buffer[esample][echannel][offsetX + x + x1][offsetY + x + y1] = tmp.buffer[esample][echannel][tmp.offsetX + x1][tmp.offsetY + y1];
                    }
                }
            }
        }
    }

    @Override
    public void setSubPanel(int x, int y, PanelElementData panel) {
        FloatPanelElementData tmp = (FloatPanelElementData) panel;
        this.setSubPanel(x, y, tmp.windowWidth, tmp.windowHeight, panel);
    }

    @Override
    public PanelElementData getElement(int x, int y) {
        return this.getSubPanel(x, y, 1, 1);
    }

    @Override
    public void setElement(int x, int y, PanelElementData panel) {
        FloatPanelElementData tmp = (FloatPanelElementData) panel;

        for (int esample = 0; esample < samples; esample++) {
            for (int echannel = 0; echannel < channels; echannel++) {
                buffer[esample][echannel][offsetX + x][offsetY + y] = tmp.buffer[esample][echannel][tmp.offsetX][tmp.offsetY];
            }
        }
    }

    @Override
    public Object getSample(int sample) {
        float[][][] output = new float[channels][windowWidth][windowHeight];
        for (int echannel = 0; echannel < channels; echannel++) {
            for (int x = 0; x < windowWidth; x++) {
                System.arraycopy(buffer[sample][echannel][offsetX + x], offsetY, output[x], 0, windowHeight);
            }
        }
        return output;
    }

    @Override
    public void setSample(int sample, Object theSample) {
        float[][][] input = (float[][][]) theSample;
        for (int echannel = 0; echannel < channels; echannel++) {
            for (int x = 0; x < windowWidth; x++) {
                System.arraycopy(input[echannel][x], 0, buffer[sample][echannel][offsetX + x], offsetY, windowHeight);
            }
        }

    }

    @Override
    public Object getChannel(int channel, int sample) {
        float[][] output = new float[windowWidth][windowHeight];
        for (int x = 0; x < windowWidth; x++) {
            System.arraycopy(buffer[sample][channel][offsetX + x], offsetY, output[x], 0, windowHeight);
        }
        return output;
    }

    @Override
    public void setChannel(int channel, int sample, Object theChannel) {
        float[][] input = (float[][]) theChannel;
        for (int x = 0; x < windowWidth; x++) {
            System.arraycopy(input[x], 0, buffer[sample][channel][offsetX + x], offsetY, windowHeight);
        }
    }

    @Override
    public Object get() {
        float[][][][] output = new float[samples][channels][windowWidth][windowHeight];
        for (int esamples = 0; esamples < samples; esamples++) {
            for (int echannel = 0; echannel < channels; echannel++) {
                for (int x = 0; x < windowWidth; x++) {
                    System.arraycopy(buffer[esamples][echannel][offsetX + x], offsetY, output[esamples][echannel][x], 0, windowHeight);
                }
            }
        }
        return output;
    }

    @Override
    public void set(Object all) {
        float[][][][] input = (float[][][][]) all;
        for (int esample = 0; esample < samples; esample++) {
            for (int echannel = 0; echannel < channels; echannel++) {
                for (int x = 0; x < windowWidth; x++) {
                    System.arraycopy(input[esample][echannel][x], 0, buffer[esample][echannel][offsetX + x], offsetY, windowHeight);
                }
            }
        }
    }

}
