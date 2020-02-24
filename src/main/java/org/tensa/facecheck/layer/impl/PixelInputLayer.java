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
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 * @param <N>
 */
public abstract class PixelInputLayer<N extends Number> implements LayerProducer<N> {
    protected BufferedImage src;
    protected NumericMatriz<N> outputLayer;
    protected boolean normalizar;
    protected boolean reflectancia;
    protected boolean preventCeroUno;
    protected boolean escalar;
    protected final List<LayerConsumer<N>> consumers;
    protected final Function<Dominio,NumericMatriz<N>> supplier;
    protected final Function<Double,N> mapper;

    public PixelInputLayer(BufferedImage src, Function<Dominio, NumericMatriz<N>> supplier, Function<Double,N> mapper, boolean normalizar) {
        this.src = src;
        this.normalizar = normalizar;
        this.consumers = new ArrayList<>();
        this.supplier = supplier;
        this.mapper = mapper;
    }

    protected NumericMatriz<N> scanInput() {
        if (src == null) {
            throw new NullPointerException("src image is null");
        }
        
        int width = src.getWidth();
        int height = src.getHeight();
        
        double[] pixels = src.getRaster().getPixels(0, 0, width, height, (double[])null);
        NumericMatriz<N> dm = supplier.apply(new Dominio(pixels.length, 1));
                
        for(int k=0;k<pixels.length;k++){
            dm.indexa(k + 1, 1, mapper.apply(pixels[k]));
        }
        
        if (preventCeroUno) {
            N escala = mapper.apply(254.0 / 255.0);
            NumericMatriz<N> margen = dm.matrizUno().productoEscalar(mapper.apply(0.5));
            dm = dm.productoEscalar(escala).adicion(margen);
        }
        if (escalar) {
            N escala = mapper.apply(1 / 255.0);
            dm = dm.productoEscalar(escala);
        }
        if (normalizar) {
            NumericMatriz<N> d = dm.distanciaE2();
            N normalizador = dm.inversoMultiplicativo(mapper.apply(Math.sqrt(d.get(Indice.D1).doubleValue())));
            dm = dm.productoEscalar(normalizador);
        }
        if (reflectancia) {
            NumericMatriz<N> r = dm.productoPunto(dm.matrizUno());
            N reflector = dm.inversoMultiplicativo(r.get(Indice.D1));
            dm = dm.productoEscalar(reflector);
        }
        return dm;
    }

    @Override
    public NumericMatriz<N> getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {
        outputLayer = scanInput();
        for (LayerConsumer<N> lc : consumers) {
            lc.seInputLayer(outputLayer);
            lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
        }
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
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
