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
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.DoubleMatriz;
import org.tensa.tensada.matrix.Indice;

/**
 *
 * @author Marcelo
 */
public class PixelsDirectOutputLayer implements LayerConsumer {
    
    private DoubleMatriz weights;
    private int status;
    private DoubleMatriz outputLayer;
    private DoubleMatriz inputLayer;
    
    private BufferedImage dest;

    public PixelsDirectOutputLayer(DoubleMatriz weights) {
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
            DoubleMatriz producto = weights.producto(inputLayer);
            DoubleMatriz distanciaE2 = (DoubleMatriz)producto.distanciaE2();
            outputLayer = (DoubleMatriz)producto
                    .productoEscalar( 1 / Math.sqrt(distanciaE2.get(Indice.D1)));
            
            DoubleMatriz pointMatriz = new DoubleMatriz(new Dominio(256, 1), DoubleStream.iterate(1.0, (i) -> i + 1.0).limit(256)
                    .boxed()
                    .collect(Collectors.toMap( d -> new Indice((int)d.intValue(), 1), d -> d-1.0)));
            DoubleMatriz pixMatriz = outputLayer.producto(pointMatriz);
            double[] pixels = new double[pixMatriz.getDominio().getFila()];
            for( int i =0; i< pixels.length; i++) {
                pixels[i] = pixMatriz.get(new Indice(i + 1, 1));
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
