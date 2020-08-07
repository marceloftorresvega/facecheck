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

import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 *
 * @author Marcelo
 * @param <N>
 */
public class SelectionInputLayer<N extends Number> implements LayerProducer<N> {

    protected final Function<Dominio,NumericMatriz<N>> supplier;
    private final LinkedList<Rectangle> areaQeue;
    private ParOrdenado idx;
    private final List<LayerConsumer<N>> consumers = new ArrayList<>();
    private NumericMatriz<N> outputLayer;

    public SelectionInputLayer(Function<Dominio, NumericMatriz<N>> supplier, LinkedList<Rectangle> areaQeue) {
        this.supplier = supplier;
        this.areaQeue = areaQeue;
    }

    public SelectionInputLayer(Function<Dominio, NumericMatriz<N>> supplier, LinkedList<Rectangle> areaQeue, ParOrdenado idx) {
        this.supplier = supplier;
        this.areaQeue = areaQeue;
        this.idx = idx;
    }

    @Override
    public NumericMatriz<N> getOutputLayer() {
        return this.outputLayer;
    }

    @Override
    public void startProduction() {
        outputLayer = supplier.apply(new Dominio(areaQeue.size(), 1));
        for (int i = 1; i <= areaQeue.size(); i++) {
            Rectangle sel = areaQeue.get(i-1);
            if (sel.contains(idx.getFila(), idx.getColumna())) {
                outputLayer.indexa(i, 1, outputLayer.getUnoValue());
            }
        }
        for (LayerConsumer<N> lc : consumers) {
            lc.setInputLayer(outputLayer);
            lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
        }
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return this.consumers;
    }

    public ParOrdenado getIdx() {
        return idx;
    }

    public void setIdx(ParOrdenado idx) {
        this.idx = idx;
    }
    
}
