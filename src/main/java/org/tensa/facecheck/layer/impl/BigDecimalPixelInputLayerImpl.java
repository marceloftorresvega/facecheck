/*
 * The MIT License
 *
 * Copyright 2020 Marcelo.
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
import java.math.BigDecimal;
import org.tensa.tensada.matrix.BigDecimalMatriz;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 *
 * @author Marcelo
 */
public class BigDecimalPixelInputLayerImpl extends PixelInputLayer<BigDecimal> {
 
    public BigDecimalPixelInputLayerImpl() {
        super(null,false);
    }
    
    public BigDecimalPixelInputLayerImpl(BufferedImage src, boolean normalizar) {
        super(src, normalizar);
    }

    @Override
    protected BigDecimal supplier(double n) {
        return BigDecimal.valueOf(n);
    }

    @Override
    protected NumericMatriz<BigDecimal> rawScan() {
        int width = src.getWidth();
        int height = src.getHeight();
        
        double[] pixels = src.getRaster().getPixels(0, 0, width, height, (double[])null);
        NumericMatriz<BigDecimal> dm = new BigDecimalMatriz(new Dominio(pixels.length, 1));
        for(int k=0;k<pixels.length;k++){
            dm.indexa(k + 1, 1, supplier(pixels[k]) );

        }        
        return dm;
    }

    @Override
    public BigDecimal activateFunction(ParOrdenado i, BigDecimal value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
