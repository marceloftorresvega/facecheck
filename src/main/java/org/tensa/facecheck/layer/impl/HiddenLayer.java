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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.activation.Activation;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.facecheck.layer.facade.SigmoidHiddenLayer;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 * @param <N>
 */
public class HiddenLayer<N extends Number> implements LayerConsumer<N>, LayerLearning<N>, LayerProducer<N> {
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
    protected final Activation<N> activation;

    public HiddenLayer(NumericMatriz<N> weights, N learningFactor, Activation<N> activation) {
        this.weights = weights;
        this.learningFactor = learningFactor;
        this.consumers = new ArrayList<>();
        this.producers = new ArrayList<>();
        this.activation = activation;
    }

    @Override
    public NumericMatriz<N> seInputLayer(NumericMatriz<N> inputLayer) {
        NumericMatriz<N> last = this.inputLayer;
        this.inputLayer = inputLayer;
        return last;
    }

    @Override
    public NumericMatriz<N> getOutputLayer() {
        return outputLayer;
    }

    @Override
    public NumericMatriz<N> getWeights() {
        return weights;
    }

    @Override
    public void layerComplete(int status) {
        this.status = status;
        if (status == LayerConsumer.SUCCESS_STATUS) {
            this.startProduction();
        }
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
            //outputLayer.replaceAll((ParOrdenado i, N v) -> 1 / (1 + Math.exp(-v.doubleValue())));
//            outputLayer = weights.producto(inputLayer);
//            activation.getActivation().apply(outputLayer);
            outputLayer = activation.getActivation()
                    .compose(weights::producto)
                    .apply(inputLayer);
            
            for (LayerConsumer<N> lc : consumers) {
                lc.seInputLayer(outputLayer);
                lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
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
            error = activation.getError().apply(learningData, outputLayer);
            //propagationError = (NumericMatriz<N>) weights.productoPunto(error);
            try (final NumericMatriz<N> punto = error.productoPunto(weights)) {
                propagationError =  punto.transpuesta();
            }
            try (final NumericMatriz<N> tensor = error.productoTensorial(inputLayer);
                    final NumericMatriz<N> delta = tensor.productoEscalar(learningFactor);
                    final NumericMatriz<N> adicion = weights.adicion(delta)) {
                synchronized (weights) {
                    weights.putAll(adicion);
                }
            }
        } catch (IOException ex) {
            log.error("startLearning", ex);
        }
        for (LayerLearning<N> back : getProducers()) {
            back.setLearningData(propagationError);
            back.startLearning();
        }
    }

    @Override
    public NumericMatriz<N> getError() {
        return error.distanciaE2().productoEscalar(error.mapper(0.5));
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
