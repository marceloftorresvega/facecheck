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
import org.tensa.tensada.matrix.Matriz;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * junta salidas capas en una sola salida
 * @author Marcelo
 * @param <N>
 */
public class BlockJoinLayer<N extends Number> implements LayerProducer<N> {


    public BlockJoinLayer(BlockMatriz<N> transportMatrix) {
        this.transportMatrix = transportMatrix;
        asignConsumer();
    }

    private final BlockMatriz<N> transportMatrix;
    private final List<LayerConsumer<N>> consumers = new ArrayList<>();
    private final List<LayerConsumer<N>> joinConsumers = new ArrayList<>();
    private final List<Indice> checkOut = new ArrayList<>();
    private NumericMatriz<N> outputLayer;
    
    @Override
    public NumericMatriz<N> getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {
        Matriz<N> builded = transportMatrix.build();
        NumericMatriz<N> id1 = (NumericMatriz<N>)transportMatrix.get(Indice.D1);
        outputLayer = id1.instancia(builded.getDominio(), builded);
        for ( LayerConsumer<N> consumer : getConsumers()) {
            consumer.setInputLayer(outputLayer);
        }
        for ( LayerConsumer<N> consumer : getConsumers()) {
            consumer.layerComplete(SUCCESS_STATUS);
        }
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return this.consumers;
    }
    
    private void asignConsumer() {
        for (int i = 1; i <= this.transportMatrix.getDominio().getFila(); i++) {
            Indice idx = new Indice(i,1);
            this.joinConsumers.add(new LocalLayerConsumerImpl(idx));
            
        }
    }

    /**
     * exporta consumidores a para recibir las entradas a unificar
     * @return List&lt;LayerConsumer&gt;
     */
    public List<LayerConsumer<N>> getJoinConsumers() {
        return joinConsumers;
    }

    private class LocalLayerConsumerImpl implements LayerConsumer<N> {

        private final Indice idx;

        public LocalLayerConsumerImpl(Indice idx) {
            this.idx = idx;
        }

        @Override
        public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
            checkOut.add(idx);
            transportMatrix.put(idx, inputLayer);
            return inputLayer;
        }

        @Override
        public void layerComplete(int status) {
            checkOut.remove(idx);
            if (checkOut.size()==0) {
                startProduction();
            }
        }
    }
}
