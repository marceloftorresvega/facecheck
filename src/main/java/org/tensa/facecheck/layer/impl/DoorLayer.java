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
import java.util.function.BooleanSupplier;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * capa que permite el paso de un resultado si y solo si se cumple una condicion
 *
 * @author Marcelo
 * @param <N>
 */
public class DoorLayer<N extends Number> implements LayerConsumer<N>, LayerProducer<N> {

    private NumericMatriz<N> inputLayer;
    private NumericMatriz<N> outputLayer;
    private List<LayerConsumer<N>> consumers = new ArrayList<>();
    private final BooleanSupplier contition;

    /**
     * constructor que incluye la condicion a comprobar
     *
     * @param contition
     */
    public DoorLayer(BooleanSupplier contition) {
        this.contition = contition;
    }

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
        if (this.contition.getAsBoolean()) {
            startProduction();
        }
    }

    @Override
    public NumericMatriz<N> getOutputLayer() {
        return this.outputLayer;
    }

    @Override
    public void startProduction() {
        outputLayer = inputLayer;
            
        for (LayerConsumer<N> lc : consumers) {
            lc.setInputLayer(outputLayer);
            lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
        }
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return this.consumers;
    }

}
