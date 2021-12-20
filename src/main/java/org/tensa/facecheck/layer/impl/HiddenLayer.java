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
import java.util.function.UnaryOperator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.activation.Activation;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 * @param <N>
 */
public class HiddenLayer<N extends Number> implements LayerConsumer<N>, LayerLearning<N>, LayerProducer<N> {
    protected final Logger log = LoggerFactory.getLogger(HiddenLayer.class);
    protected final NumericMatriz<N> weights;
    protected int status;
    protected NumericMatriz<N> outputLayer;
    protected NumericMatriz<N> inputLayer;
    protected NumericMatriz<N> propagationError;
    protected NumericMatriz<N> learningData;
    protected NumericMatriz<N> error;
    protected NumericMatriz<N> net;
    protected N learningFactor;
    protected final List<LayerConsumer<N>> consumers;
    protected final List<LayerLearning<N>> producers;
    protected final Activation<N> activation;
    protected final boolean useBias;

    public HiddenLayer(NumericMatriz<N> weights, N learningFactor, Activation<N> activation, boolean useBias) {
        this.weights = weights;
        this.learningFactor = learningFactor;
        this.consumers = new ArrayList<>();
        this.producers = new ArrayList<>();
        this.activation = activation;
        this.useBias = useBias;
    }

    public HiddenLayer(NumericMatriz<N> weights, Activation<N> activation) {
        this.weights = weights;
        this.consumers = new ArrayList<>();
        this.producers = new ArrayList<>();
        this.activation = activation;
        this.useBias = false;
    }

    @Override
    public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
        NumericMatriz<N> last = this.inputLayer;
        if (this.useBias) {
            Dominio dominio = inputLayer.getDominio();
            Integer fila = dominio.getFila();
            Integer columna = dominio.getColumna();
            fila = fila + 1;
            this.inputLayer = inputLayer.instancia(new Dominio(fila, columna), inputLayer);
            for (int c= 1; c <= columna ; c++) {
                this.inputLayer.indexa(fila , c, inputLayer.getUnoValue());
            }
            
        } else {
            this.inputLayer = inputLayer;
        }
            
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
            UnaryOperator<NumericMatriz<N>> assign = (n) -> net = n;
            outputLayer = assign.compose(weights::producto)
                    .andThen(activation.getActivation())
                    .apply(inputLayer);
            
            for (LayerConsumer<N> lc : consumers) {
                lc.setInputLayer(outputLayer);
            }
            
            for (LayerConsumer<N> lc : consumers) {
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
            error = activation.getError()
                    .apply(learningData, activation.isOptimized()?outputLayer:net);
             
            if ( getProducers().size() > 0 ) {
                propagationError = weights.productoPunto(error);
                if (this.useBias) {
                    Dominio dominio = propagationError.getDominio();
                    Integer fila = dominio.getFila();
                    Integer columna = dominio.getColumna();
                    fila = fila -1;
                    propagationError = propagationError.instancia(new Dominio(fila, columna), propagationError);
                    for ( int c = 1; c< columna;c++){
                        propagationError.remove(new Dominio(fila, c));
                    }
                }
            }
            
            try (
                    final NumericMatriz<N> derror = error.productoEscalar(learningFactor);
                    final NumericMatriz<N> delta = derror.productoTensorial(inputLayer);
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
        }
        for (LayerLearning<N> back : getProducers()) {
            back.startLearning();
        }
    }

    @Override
    public NumericMatriz<N> getError() {
        NumericMatriz<N> error2 = null;
        try (NumericMatriz<N> distanciaE2 = error.distanciaE2()) {
            error2 = distanciaE2.productoEscalar(error.mapper(0.5));
        } catch (IOException ex) {
            log.error("getError", ex);
        }
        return error2;
    }

    @Override
    public N getLeanringFactor() {
        return learningFactor;
    }

    @Override
    public void setLearningFactor(N learningFactor) {
        this.learningFactor = learningFactor;
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
