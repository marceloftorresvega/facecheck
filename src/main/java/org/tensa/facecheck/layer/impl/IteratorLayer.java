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
import java.util.function.BooleanSupplier;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * iterador en base a la condicion
 *
 * @author Marcelo
 * @param <N>
 */
public class IteratorLayer<N extends Number> implements LayerConsumer<N>, LayerProducer<N>, LayerLearning<N> {

    private final BooleanSupplier condition;
    private final boolean nullLearning;
    private NumericMatriz<N> outputLayer;
    private NumericMatriz<N> inputLayer;
    private int status;
    private final List<LayerConsumer<N>> bindProducerConsumers = new ArrayList<>();
    private final List<LayerLearning<N>> bindLearningBackProducers = new ArrayList<>();
    private final List<LayerLearning<N>> bindLearningBackConsumer = new ArrayList<>();
    private NumericMatriz<N> bindLearningProducerLearningData;
    private NumericMatriz<N> bindLearningConsumerLearningData;
    private final List<LayerConsumer<N>> consumers = new ArrayList<>();

    private final LayerProducer<N> bindProducer = new LayerProducer<N>() {
        @Override
        public NumericMatriz<N> getOutputLayer() {
            return inputLayer;
        }

        @Override
        public void startProduction() {

            for (LayerConsumer<N> lc : bindProducerConsumers) {
                lc.setInputLayer(inputLayer);
            }

            for (LayerConsumer<N> lc : bindProducerConsumers) {
                lc.layerComplete(status);
            }
        }

        @Override
        public List<LayerConsumer<N>> getConsumers() {
            return bindProducerConsumers;
        }
    };

    private final LayerConsumer<N> bindConsumer = new LayerConsumer<N>() {
        @Override
        public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
            NumericMatriz<N> old = outputLayer;
            outputLayer = inputLayer;
            return old;
        }

        @Override
        public void layerComplete(int status) {
            //
        }
    };

    private final LayerLearning<N> bindLearningProducer = new LayerLearning<N>() {
        @Override
        public NumericMatriz<N> getWeights() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public NumericMatriz<N> getPropagationError() {
            return bindLearningProducerLearningData;
        }

        @Override
        public NumericMatriz<N> getError() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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
            bindLearningProducerLearningData = learningData;
        }

        @Override
        public void startLearning() {

            for (LayerLearning<N> back : getProducers()) {
                back.setLearningData(bindLearningProducerLearningData);
            }
            for (LayerLearning<N> back : getProducers()) {
                back.startLearning();
            }
        }

        @Override
        public List<LayerLearning<N>> getProducers() {
            return bindLearningBackProducers;
        }
    };

    private final LayerLearning<N> bindLearningConsumer = new LayerLearning<N>() {
        @Override
        public NumericMatriz<N> getWeights() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public NumericMatriz<N> getPropagationError() {
            return bindLearningConsumerLearningData;
        }

        @Override
        public NumericMatriz<N> getError() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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
            bindLearningConsumerLearningData = learningData;
        }

        @Override
        public void startLearning() {
            for (LayerLearning<N> back : getProducers()) {
                back.setLearningData(bindLearningConsumerLearningData);
            }
            for (LayerLearning<N> back : getProducers()) {
                back.startLearning();
            }
        }

        @Override
        public List<LayerLearning<N>> getProducers() {
            return bindLearningBackConsumer;
        }
    };

    public IteratorLayer(BooleanSupplier condition, boolean nullLearning) {
        this.condition = condition;
        this.nullLearning = nullLearning;
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
    public NumericMatriz<N> getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {
        if (status == LayerConsumer.SUCCESS_STATUS) {
            while (condition.getAsBoolean()) {
                bindProducer.startProduction();
                if (nullLearning) {
                    this.bindLearningConsumer.startLearning();
                }
            }

            for (LayerConsumer<N> lc : consumers) {
                lc.setInputLayer(outputLayer);
            }

            for (LayerConsumer<N> lc : consumers) {
                lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
            }
        }
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return this.consumers;
    }

    @Override
    public NumericMatriz<N> getWeights() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public NumericMatriz<N> getPropagationError() {
        return bindLearningProducerLearningData;
    }

    @Override
    public NumericMatriz<N> getError() {
        NumericMatriz<N> error2 = null;
        try (NumericMatriz<N> distanciaE2 = bindLearningProducerLearningData.distanciaE2()) {
            error2 = distanciaE2.productoEscalar(bindLearningProducerLearningData.mapper(0.5));
        } catch (IOException ex) {
            //
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
        bindLearningConsumer.setLearningData(learningData);
    }

    @Override
    public void startLearning() {
        if (!nullLearning) {
            while (condition.getAsBoolean()) {
                bindLearningConsumer.startLearning();
            }
        }
    }

    @Override
    public List<LayerLearning<N>> getProducers() {
        return bindLearningBackProducers;
    }

    public LayerProducer<N> getBindProducer() {
        return bindProducer;
    }

    public LayerConsumer<N> getBindConsumer() {
        return bindConsumer;
    }

    public LayerLearning<N> getBindLearningProducer() {
        return bindLearningProducer;
    }

    public LayerLearning<N> getBindLearningConsumer() {
        return bindLearningConsumer;
    }

}
