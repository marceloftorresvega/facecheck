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

import org.tensa.facecheck.layer.facade.LinealLeanringLayer;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 *
 * @author lorenzo
 * @param <N>
 */
public abstract class LearningLayer<N extends Number> implements LayerConsumer<N>, LayerLearning<N>, LayerProducer<N> {
    protected final Logger log = LoggerFactory.getLogger(LinealLeanringLayer.class);
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

    public LearningLayer(NumericMatriz<N> weights, N learningFactor) {
        this.weights = weights;
        this.learningFactor = learningFactor;
        this.consumers = new ArrayList<>();
        this.producers = new ArrayList<>();
    }


    @Override
    public NumericMatriz<N> seInputLayer(NumericMatriz<N> inputLayer) {
        this.inputLayer = inputLayer;
        return inputLayer;
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
            this.calculateErrorOperation();
            //        propagationError = (NumericMatriz<N>) weights.productoPunto(error);
            try (final NumericMatriz<N> punto = error.productoPunto(weights)) {
                propagationError = (NumericMatriz<N>) punto.transpuesta();
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
        for (LayerLearning back : getProducers()) {
            back.setLearningData(propagationError);
            back.startLearning();
        }
        producers.clear();
    }

    @Override
    public NumericMatriz<N> getError() {
        if (error != null) {
            return error.distanciaE2().productoEscalar(mediaError(0.5));
        } else {
            return supplier();
        }
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
    public NumericMatriz<N> getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {
        outputLayer = weights.producto(inputLayer);
        outputLayer.replaceAll(this::activateFunction);
        
        for (LayerConsumer lc : consumers) {
            lc.seInputLayer(outputLayer);
            lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
            if (lc instanceof LayerLearning) {
                ((LayerLearning) lc).getProducers().add(this);
            }
        }
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return consumers;
    }

    @Override
    public abstract void calculateErrorOperation();

    @Override
    public abstract N activateFunction(ParOrdenado i, N value);

    public abstract N mediaError(double v);
    
    protected abstract NumericMatriz<N> supplier();
    
}
