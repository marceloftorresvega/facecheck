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
import java.util.ArrayList;
import java.util.List;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.DoubleMatriz;

/**
 *
 * @author Marcelo
 */
public class SimplePixelsDirectInputLayer extends ArrayList<LayerConsumer> implements LayerProducer {
    
    private BufferedImage src;
    private DoubleMatriz outputLayer;
    
    private DoubleMatriz scanInput(){
        
        if (src == null) {
            throw new NullPointerException("src image is null");
        }
        
        int width = src.getWidth();
        int height = src.getHeight();
        
        double[] pixels = src.getRaster().getPixels(0, 0, width, height, (double[])null);
        DoubleMatriz dm = new DoubleMatriz(new Dominio(pixels.length, 1));
        for(int k=0;k<pixels.length;k++){
            dm.indexa(k + 1, 1, pixels[k] * 254 / 255 + 0.5);

        }
//        NumericMatriz<Double> d = dm.distanciaE2();
//        d.replaceAll((k, v) -> 1/ Math.sqrt(v));
//        
//
//        return (DoubleMatriz)d.productoKronecker(dm);
        return dm;
    }

    @Override
    public DoubleMatriz getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {
        outputLayer = scanInput();
        for( LayerConsumer lc : this){
            lc.seInputLayer(outputLayer);
            lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
        }
    }

    @Override
    public List<LayerConsumer> getConsumers() {
        return this;
    }

    public void setSrc(BufferedImage src) {
        this.src = src;
    }
    
}
