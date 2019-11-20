/*
 * The MIT License
 *
 * Copyright 2019 Marcelo.
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
package org.tensa.facecheck.layer.impl;

import java.awt.image.BufferedImage;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.tensada.matrix.DoubleMatriz;
import org.tensa.tensada.matrix.Indice;

/**
 *
 * @author Marcelo
 */
public class PixelsLinealDirectOutputLayer implements LayerConsumer {
    
    private final DoubleMatriz weights;
    private int status;
    private DoubleMatriz outputLayer;
    private DoubleMatriz inputLayer;
    
    private BufferedImage dest;

    public PixelsLinealDirectOutputLayer(DoubleMatriz weights) {
        this.weights = weights;
    }

    @Override
    public DoubleMatriz seInputLayer(DoubleMatriz inputLayer) {
        this.inputLayer = inputLayer;
        return inputLayer;
    }

    @Override
    public DoubleMatriz getWeights() {
        return weights;
    }

    @Override
    public void layerComplete(int status) {
        this.status = status;
        if (status == LayerConsumer.SUCCESS_STATUS) {
            DoubleMatriz distanciaE2 = (DoubleMatriz)inputLayer.distanciaE2();
            outputLayer = (DoubleMatriz)inputLayer
                    .productoEscalar( 255 / Math.sqrt(distanciaE2.get(Indice.D1)));
            
            double[] pixels = new double[outputLayer.getDominio().getFila()];
            for( int i =0; i< pixels.length; i++) {
                pixels[i] = outputLayer.get(new Indice(i + 1, 1));
            }
            
            int width = dest.getWidth();
            int height = dest.getHeight();
            dest.getRaster().setPixels(0, 0, width, height, pixels);
            
        }
    }

    public BufferedImage getDest() {
        return dest;
    }

    public void setDest(BufferedImage dest) {
        this.dest = dest;
    }

}
