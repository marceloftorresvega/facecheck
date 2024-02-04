/*
 * The MIT License
 *
 * Copyright 2021 Marcelo.
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
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.activation.Activation;
import org.tensa.facecheck.activation.utils.ActivationUtils;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 * clasificador de caractereisticas competitivo
 *
 * @author Marcelo
 * @param <N>
 */
public class CpnLayer<N extends Number> implements LayerConsumer<N>, LayerLearning<N>, LayerProducer<N> {

    protected final Logger log = LoggerFactory.getLogger(HiddenLayer.class);
    protected final NumericMatriz<N> weights;
    protected int status;
    protected NumericMatriz<N> outputLayer;
    protected NumericMatriz<N> inputLayer;
    protected NumericMatriz<N> propagationError;
    protected NumericMatriz<N> learningData;
    protected NumericMatriz<N> error;
    protected N learningFactor;
    protected final List<LayerConsumer<N>> consumers;
    protected final List<LayerLearning<N>> producers;
    protected final Activation<N> activation;

    public CpnLayer(NumericMatriz<N> weights, Activation<N> activation) {
        this.activation = activation;
        this.producers = new ArrayList<>();
        this.consumers = new ArrayList<>();
        this.weights = weights;
    }

    public CpnLayer(NumericMatriz<N> weights, N learningFactor, Activation<N> activation) {
        this.activation = activation;
        this.producers = new ArrayList<>();
        this.consumers = new ArrayList<>();
        this.weights = weights;
        this.learningFactor = learningFactor;
    }

    @Override
    public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
        NumericMatriz<N> old = this.inputLayer;
        this.inputLayer = inputLayer;
        return old;
    }

    @Override
    public void layerComplete(int status) {
        this.status = status;
        this.startProduction();
    }

    @Override
    public NumericMatriz<N> getWeights() {
        return weights;
    }

    @Override
    public NumericMatriz<N> getPropagationError() {
        return propagationError;
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
    public void setLearningData(NumericMatriz<N> learningData) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void startLearning() {
        try (
                final NumericMatriz<N> seleccion = weights.productoPunto(outputLayer);) {
            propagationError = inputLayer.substraccion(seleccion);
            try (
                    final NumericMatriz<N> amplif = outputLayer.productoEscalar(learningFactor);
                    final NumericMatriz<N> delta = amplif.productoTensorial(propagationError);
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
    public List<LayerLearning<N>> getProducers() {
        return producers;
    }

    @Override
    public NumericMatriz<N> getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {
        if (status == LayerConsumer.SUCCESS_STATUS) {
            Integer columna = inputLayer.getDominio().getColumna();
            Integer fila = weights.getDominio().getFila();
            Dominio dominio = new Dominio(fila, columna);

            outputLayer = activation.getActivation()
                    .compose(this::getDiferencia)
                    .apply(dominio);

            for (LayerConsumer<N> lc : consumers) {
                lc.setInputLayer(outputLayer);
            }

            for (LayerConsumer<N> lc : consumers) {
                lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
            }
        }
    }

    private NumericMatriz<N> getDiferencia(Dominio d) {
        Integer barrida
                = weights.getDominio().getColumna();
        return error = d.stream()
                .collect(
                        ActivationUtils.domainToMatriz(
                                () -> inputLayer.instancia(d),
                                (ParOrdenado k) -> {
                                    return IntStream.rangeClosed(1, barrida).mapToObj(j -> {
                                        Indice ka = new Indice(k.getFila(), j);
                                        Indice kb = new Indice(j, k.getColumna());
                                        N resta = weights.resta(inputLayer.get(kb), weights.get(ka));
                                        return weights.multiplica(resta, resta);
                                    }).collect(
                                            Collectors.collectingAndThen(
                                                            Collectors.summingDouble(N::doubleValue),
                                                    weights::mapper)
                                    );
                                }));
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return consumers;
    }

}
