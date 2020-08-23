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
import static org.tensa.facecheck.layer.LayerConsumer.SUCCESS_STATUS;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.BlockMatriz;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * separa salida de una capa en varias
 * @author Marcelo
 * @param <N>
 */
public class BlockSplitLayer<N extends Number> implements LayerConsumer<N> {

    public BlockSplitLayer(BlockMatriz<N> transportMatrix) {
        this.transportMatrix = transportMatrix;
        asignProducers();
    }

    private final BlockMatriz<N> transportMatrix;
    private final List<LayerProducer<N>> splitProducers = new ArrayList<>();
    private NumericMatriz<N> inputLayer;

    @Override
    public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
        return this.inputLayer=inputLayer;
    }

    @Override
    public NumericMatriz<N> getWeights() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void layerComplete(int status) {
        transportMatrix.splitIn(inputLayer);
        for ( LayerProducer<N> producer : splitProducers ) {
            producer.startProduction();
        }
    }
    
    private void asignProducers() {
        for (int i = 1; i <= this.transportMatrix.getDominio().getFila(); i++) {
            Indice idx = new Indice(i,1);
            this.splitProducers.add(new LocalLayerProducerImpl(idx));
            
        }
    }

    /**
     * expone producers que distribuyen las salidas producidas
     * @return List&lt;LayerProducer&gt;
     */
    public List<LayerProducer<N>> getSplitProducers() {
        return splitProducers;
    }

    private class LocalLayerProducerImpl implements LayerProducer<N> {

        private final Indice idx;
        private final List<LayerConsumer<N>> localConsumers  = new ArrayList<>();

        public LocalLayerProducerImpl(Indice idx) {
            this.idx = idx;
        }

        @Override
        public NumericMatriz<N> getOutputLayer() {
            return (NumericMatriz<N>)transportMatrix.get(idx);
        }

        @Override
        public void startProduction() {
            for ( LayerConsumer<N> consumer : getConsumers()) {
                consumer.setInputLayer(getOutputLayer());
            }
            for ( LayerConsumer<N> consumer : getConsumers()) {
                consumer.layerComplete(SUCCESS_STATUS);
            }
            
        }

        @Override
        public List<LayerConsumer<N>> getConsumers() {
            return localConsumers;
        }
    }
    
}
