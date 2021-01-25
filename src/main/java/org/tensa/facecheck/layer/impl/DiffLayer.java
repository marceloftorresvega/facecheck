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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * diff layer compara unretorno de una capa producer con otra y retorna la
 * diferencia como errror para back
 *
 * @author Marcelo
 * @param <N>
 */
public class DiffLayer<N extends Number> implements LayerLearning<N> {

    @Override
    public NumericMatriz<N> getWeights() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    /**
     * en caso de error se usa para dejar registro
     */
    protected final Logger log = LoggerFactory.getLogger(HiddenLayer.class);

    /**
     * instancia capa diff con capa de comparacion (valor deseado) y adjunta un
     * error consumer para informar del avance
     *
     * @param compareLayer valor deseado
     * @param errorConsumer consumer que informa del error
     */
    public DiffLayer(LayerProducer<N> compareLayer, Consumer<LayerLearning<N>> errorConsumer) {
        this.internalBridgeConsumer = new LayerConsumer<N>() {

            @Override
            public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
                salidaReal = inputLayer;
                return inputLayer;
            }

            @Override
            public void layerComplete(int status) {
                if( compareLayer instanceof LayerConsumer) {
                    ((LayerConsumer)compareLayer).setInputLayer(salidaReal);
                    ((LayerConsumer)compareLayer).layerComplete(status);
                } else {
                    compareLayer.startProduction();
                }
                    
            }
        };
        this.internalBackConsumer = new LayerConsumer<N>() {
            @Override
            public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
                setLearningData(inputLayer);
                return inputLayer;
            }

            @Override
            public void layerComplete(int status) {
                startLearning();
            }
        };
        this.compareLayer = compareLayer;
        this.compareLayer.getConsumers().add(internalBackConsumer);
        this.errorConsumer = errorConsumer;
    }

    private final LayerProducer<N> compareLayer;
    private NumericMatriz<N> salidaReal;
    private NumericMatriz<N> salidaDeseada;
    private final List<LayerLearning<N>> producers = new ArrayList<>();
    private NumericMatriz<N> propagationError;
    private final LayerConsumer<N> internalBackConsumer;
    private final LayerConsumer<N> internalBridgeConsumer;
    private final Consumer<LayerLearning<N>> errorConsumer;

    @Override
    public NumericMatriz<N> getPropagationError() {
        return propagationError;
    }

    @Override
    public NumericMatriz<N> getError() {
        NumericMatriz<N> error2 = null;
        try (NumericMatriz<N> distanciaE2 = propagationError.distanciaE2()) {
            error2 = distanciaE2.productoEscalar(propagationError.mapper(0.5));
        } catch (IOException ex) {
            log.error("getError", ex);
        }
        return error2;
    }

    @Override
    public N getLeanringFactor() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setLearningFactor(N learningFactor) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setLearningData(NumericMatriz<N> learningData) {
        salidaDeseada = learningData;
    }

    @Override
    public void startLearning() {
        propagationError = salidaDeseada.substraccion(salidaReal);
        Optional.ofNullable(errorConsumer).ifPresent(cnsmr -> cnsmr.accept(this));
        for (LayerLearning<N> back : producers) {
            back.setLearningData(propagationError);
            back.startLearning();
        }
    }

    @Override
    public List<LayerLearning<N>> getProducers() {
        return producers;
    }

    /**
     * puente interno layer consumer punto de enlace con el resto de la red y
     * por donde ingrresa el valor calculado a comparar
     *
     * @return LayerConsumer
     */
    public LayerConsumer<N> getInternalBridgeConsumer() {
        return internalBridgeConsumer;
    }

}
