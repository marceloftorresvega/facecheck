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
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 */
public class PixelsDirectInputLayer implements LayerProducer {
    
    private BufferedImage src;
    private DoubleMatriz outputLayer;
    private boolean normalizar;
    private boolean reflectancia;
    private boolean preventCeroUno;
    private boolean escalar;
    private final List<LayerConsumer> consumers;

    public PixelsDirectInputLayer() {
        this.consumers = new ArrayList<>();
    }

    public PixelsDirectInputLayer(BufferedImage src, boolean normalizar) {
        this.src = src;
        this.normalizar = normalizar;
        this.consumers = new ArrayList<>();
    }
    
    private DoubleMatriz scanInput(){
        
        if (src == null) {
            throw new NullPointerException("src image is null");
        }
        
        int width = src.getWidth();
        int height = src.getHeight();
        
        double[] pixels = src.getRaster().getPixels(0, 0, width, height, (double[])null);
        DoubleMatriz dm = new DoubleMatriz(new Dominio(pixels.length, 1));
        for(int k=0;k<pixels.length;k++){
            dm.indexa(k + 1, 1, pixels[k] );

        }
        if(preventCeroUno) {
            double escala = 254.0/255.0;
            NumericMatriz<Double> margen = dm.matrizUno().productoEscalar(0.5);
            dm = (DoubleMatriz)dm.productoEscalar(escala).adicion(margen);
        }
        if(escalar) {
            double escala = 1/255.0;
            dm = (DoubleMatriz)dm.productoEscalar(escala);
        }
        if(normalizar) {
            NumericMatriz<Double> d = dm.distanciaE2();
            double normalizador = 1/ Math.sqrt( d.get(Indice.D1));
            dm = (DoubleMatriz)dm.productoEscalar(normalizador);
        }
        if(reflectancia) {
            NumericMatriz<Double> r = dm.productoPunto(dm.matrizUno());
            double reflector =  1/ r.get(Indice.D1);
            dm = (DoubleMatriz)dm.productoEscalar(reflector);
        }
        
        return dm;
    }

    @Override
    public DoubleMatriz getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {
        outputLayer = scanInput();
        for( LayerConsumer lc : consumers){
            lc.seInputLayer(outputLayer);
            lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
        }
    }

    @Override
    public List<LayerConsumer> getConsumers() {
        return consumers;
    }

    public void setSrc(BufferedImage src) {
        this.src = src;
    }

    public boolean isNormalizar() {
        return normalizar;
    }

    public void setNormalizar(boolean normalizar) {
        this.normalizar = normalizar;
        this.reflectancia = !normalizar;
    }

    public boolean isReflectancia() {
        return reflectancia;
    }

    public void setReflectancia(boolean reflectancia) {
        this.reflectancia = reflectancia;
        this.normalizar = !reflectancia;
    }

    public boolean isPreventCeroUno() {
        return preventCeroUno;
    }

    public void setPreventCeroUno(boolean preventCeroUno) {
        this.preventCeroUno = preventCeroUno;
    }

    public boolean isEscalar() {
        return escalar;
    }

    public void setEscalar(boolean escalar) {
        this.escalar = escalar;
    }
    
}
