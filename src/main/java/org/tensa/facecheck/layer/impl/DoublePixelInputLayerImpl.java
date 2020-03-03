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
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.DoubleMatriz;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;


public class DoublePixelInputLayerImpl extends PixelInputLayer<Double> {

    public DoublePixelInputLayerImpl() {
        super(null, false);
    }

    public DoublePixelInputLayerImpl(BufferedImage src, boolean normalizar) {
        super(src, normalizar);
    }

    @Override
    protected NumericMatriz<Double> rawScan() {
        int width = src.getWidth();
        int height = src.getHeight();
        
        double[] pixels = src.getRaster().getPixels(0, 0, width, height, (double[])null);
        NumericMatriz<Double> dm = new DoubleMatriz(new Dominio(pixels.length, 1));
        for(int k=0;k<pixels.length;k++){
            dm.indexa(k + 1, 1, pixels[k] );

        }        
        return dm;
    }

    @Override
    public Double activateFunction(ParOrdenado i, Double value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected Double supplier(double n) {
        return n;
    }
    
}
