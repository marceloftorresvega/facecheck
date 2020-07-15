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
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * Capa que retorna una version normalizada de la matriz de ingreso, ha sido
 * pensada para matrices uni dimensionales (vectores)
 *
 * @author Marcelo
 * @param <N>
 */
public class NormalizeLayer<N extends Number> implements LayerConsumer<N>, LayerProducer<N> {

    private NumericMatriz<N> inputLayer;
    private NumericMatriz<N> outputLayer;
    private final List<LayerConsumer<N>> layerConsumers = new ArrayList<>();

    @Override
    public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
        this.inputLayer = inputLayer;
        return inputLayer;
    }

    @Override
    public NumericMatriz<N> getWeights() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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
        NumericMatriz<N> norma = this.inputLayer.norma();
        outputLayer = inputLayer.productoEscalar(inputLayer.inversoMultiplicativo(norma.get(Indice.D1)));
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return this.layerConsumers;
    }

}
