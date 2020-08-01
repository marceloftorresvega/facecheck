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

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.facecheck.mapping.IndexMapper;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 * input layer que ingresa a la red el indice de una convoulcion
 *
 * @author Marcelo
 */
public class IndexInputLayer<N extends Number> implements LayerProducer<N> {

    private final List<LayerConsumer<N>> consumers = new ArrayList<>();
    private final Function<Dominio,NumericMatriz<N>> matrizSuplier;
    private final IndexMapper<N> idMapper;
    private NumericMatriz<N> outputLayer;
    private ParOrdenado idx;

    public IndexInputLayer(Function<Dominio, NumericMatriz<N>> matrizSuplier, IndexMapper<N> idMapper, ParOrdenado idx) {
        this.matrizSuplier = matrizSuplier;
        this.idMapper = idMapper;
        this.idx = idx;
    }

    public IndexInputLayer(Function<Dominio,NumericMatriz<N>> matrizSupplier, IndexMapper<N> idMapper) {
        this.matrizSuplier = matrizSupplier;
        this.idMapper = idMapper;
    }

    @Override
    public NumericMatriz<N> getOutputLayer() {
        return this.outputLayer;
    }

    @Override
    public void startProduction() {
        this.outputLayer = idMapper.apply(idx, matrizSuplier.apply(idMapper.getDominio()));

    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return this.consumers;
    }

    /**
     * indice que se procesa
     *
     * @return ParOrdenado
     */
    public ParOrdenado getIdx() {
        return idx;
    }

    /**
     * alimenta el indice a procesar
     *
     * @param idx indice
     */
    public void setIdx(ParOrdenado idx) {
        this.idx = idx;
    }

}
