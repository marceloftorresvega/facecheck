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

import org.tensa.facecheck.layer.facade.SigmoidHiddenLayer;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 * @param <N>
 */
public abstract class HiddenLayer<N extends Number> implements LayerConsumer<N>, LayerLearning<N>, LayerProducer<N> {
    protected final Logger log = LoggerFactory.getLogger(SigmoidHiddenLayer.class);
    protected final NumericMatriz<N> weights;
    protected int status;
    protected NumericMatriz<N> outputLayer;
    protected NumericMatriz<N> inputLayer;
    protected NumericMatriz<N> propagationError;
    protected NumericMatriz<N> learningData;
    protected NumericMatriz<N> error;
    protected final N learningFactor;
    protected final List<LayerConsumer<N>> consumers;
    protected final List<LayerLearning<N>> producers;


    public HiddenLayer(NumericMatriz<N> weights, N learningFactor) {
        this.weights = weights;
        this.learningFactor = learningFactor;
        this.consumers = new ArrayList<>();
        this.producers = new ArrayList<>();
    }

    @Override
    public NumericMatriz<N> seInputLayer(NumericMatriz<N> inputLayer) {
        this.inputLayer = inputLayer;
        return null;
    }

    @Override
    public NumericMatriz<N> getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void layerComplete(int status) {
        this.status = status;
        this.startProduction();
    }

    @Override
    public void startProduction() {
        if (status == LayerConsumer.SUCCESS_STATUS) {
            //            log.info("pesos <{}><{}>", weights.getDominio().getFila(), weights.getDominio().getColumna());
            //            log.info("layer <{}><{}>", inputLayer.getDominio().getFila(), inputLayer.getDominio().getColumna());
            //            NumericMatriz<N> producto = weights.producto(inputLayer);
            //            NumericMatriz<N> distanciaE2 = (NumericMatriz<N>)producto.distanciaE2();
            //            outputLayer = (NumericMatriz<N>)producto
            //                    .productoEscalar( 1 / Math.sqrt(distanciaE2.get(Indice.D1)));
            outputLayer = weights.producto(inputLayer);
            outputLayer.replaceAll(this::activateFunction);
            //outputLayer.replaceAll((ParOrdenado i, N v) -> 1 / (1 + Math.exp(-v.doubleValue())));
            
            for (LayerConsumer lc : consumers) {
                lc.seInputLayer(outputLayer);
                lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
                if (lc instanceof LayerLearning) {
                    ((LayerLearning) lc).getProducers().add(this);
                }
            }
        }
    }

    @Override
    public NumericMatriz<N> getPropagationError() {
        return propagationError;
    }

    @Override
    public void setLearningData(NumericMatriz<N> learningData) {
        this.learningData = learningData;
    }

    @Override
    public void startLearning() {
        try {
//            
//            try (final NumericMatriz<N> m1 = outputLayer.matrizUno()) {
//                error = (NumericMatriz<N>) m1.substraccion(outputLayer);
//                error.replaceAll((ParOrdenado i, N v) -> outputLayer.productoDirecto(outputLayer.productoDirecto(v, outputLayer.get(i)), learningData.get(i)));
//            }
            this.calculateErrorOperation();
            
            //        propagationError = (NumericMatriz<N>) weights.productoPunto(error);
            try (final NumericMatriz<N> punto = error.productoPunto(weights)) {
                propagationError = (NumericMatriz<N>) punto.transpuesta();
            }
            try (final NumericMatriz<N> tensor = error.productoTensorial(inputLayer);final NumericMatriz<N> delta = tensor.productoEscalar(learningFactor);final NumericMatriz<N> adicion = weights.adicion(delta)) {
                synchronized (weights) {
                    weights.putAll(adicion);
                }
            }
        } catch (IOException ex) {
            log.error("startLearning", ex);
        }
        for (LayerLearning back : getProducers()) {
            back.setLearningData(propagationError);
            back.startLearning();
        }
        getProducers().clear();
    }

    @Override
    public NumericMatriz<N> getWeights() {
        return weights;
    }

    @Override
    public NumericMatriz<N> getError() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public N getLeanringFactor() {
        return learningFactor;
    }

    @Override
    public List<LayerLearning<N>> getProducers() {
        return producers;
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return consumers;
    }
    
}
