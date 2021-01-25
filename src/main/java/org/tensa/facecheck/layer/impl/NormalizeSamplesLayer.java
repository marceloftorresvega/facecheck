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
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.BlockMatriz;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 * Capa que retorna una version normalizada de las matrices columnas de la
 * matriz, hecha pensando en las muestras de ingreso
 *
 * @author Marcelo
 * @param <N>
 */
public class NormalizeSamplesLayer<N extends Number> implements LayerConsumer<N>, LayerProducer<N> {

    private NumericMatriz<N> inputLayer;
    private NumericMatriz<N> outputLayer;
    private final List<LayerConsumer<N>> layerConsumers = new ArrayList<>();

    @Override
    public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
        this.inputLayer = inputLayer;
        return inputLayer;
    }

    @Override
    public void layerComplete(int status) {
        startProduction();
    }

    @Override
    public NumericMatriz<N> getOutputLayer() {
        return this.outputLayer;
    }

    @Override
    public void startProduction() {
        Dominio dominio = inputLayer.getDominio();
        Integer columnas = dominio.getColumna();
        Integer filas = dominio.getFila();

        BlockMatriz<N> samples = new BlockMatriz<>(new Dominio(1, columnas));
        samples.getDominio().forEach((ParOrdenado idx) -> {
            samples.put(idx, inputLayer.instancia(new Dominio(filas, 1)));
        });
        samples.splitIn(inputLayer);
        samples.replaceAll((idx, matriz) -> OutputScale.normalized((NumericMatriz<N>) matriz));
        outputLayer = inputLayer.instancia(dominio, samples.build());
        samples.clear();
            
        for (LayerConsumer<N> lc : layerConsumers) {
            lc.setInputLayer(outputLayer);
            lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
        }
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return this.layerConsumers;
    }

}
