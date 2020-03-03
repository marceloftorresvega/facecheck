/*
 * The MIT License
 *
 * Copyright 2020 lorenzo.
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
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author lorenzo
 * @param <N>
 */
public abstract class PixelOutputLayer<N extends Number> implements LayerConsumer<N> {
    protected int status;
    protected NumericMatriz<N> inputLayer;
    protected BufferedImage dest;

    public PixelOutputLayer() {
    }

    

    @Override
    public NumericMatriz<N> seInputLayer(NumericMatriz<N> inputLayer) {
        this.inputLayer = inputLayer;
        return inputLayer;
    }

    @Override
    public NumericMatriz<N> getWeights() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }


    @Override
    public abstract void layerComplete(int status);

    public BufferedImage getDest() {
        return dest;
    }

    public void setDest(BufferedImage dest) {
        this.dest = dest;
    }
    
}
