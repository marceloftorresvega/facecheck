/*
 * The MIT License
 *
 * Copyright 2019 Marcelo.
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.DoubleMatriz;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.facecheck.layer.LayerLearning;

/**
 *
 * @author Marcelo
 */
public class PixelDirectLinealLeanringLayer implements LayerConsumer, LayerLearning, LayerProducer {
    
    private final Logger log = LoggerFactory.getLogger(PixelDirectLinealLeanringLayer.class);
    
    private final DoubleMatriz weights;
    private int status;
    private DoubleMatriz outputLayer;
    private DoubleMatriz inputLayer;
    
    private DoubleMatriz propagationError;
    private DoubleMatriz learningData;
    private DoubleMatriz error;
    private final Double learningFactor;
    private final List<LayerConsumer> consumers;
    private final List<LayerLearning> producers;

    public PixelDirectLinealLeanringLayer(DoubleMatriz weights, Double learningStep) {
        this.weights = weights;
        this.learningFactor = learningStep;
        this.consumers = new ArrayList<>();
        this.producers = new ArrayList<>();
    }

    @Override
    public DoubleMatriz seInputLayer(DoubleMatriz inputLayer) {
        this.inputLayer = inputLayer;
        return inputLayer;
    }

    @Override
    public DoubleMatriz getWeights() {
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
    public DoubleMatriz getPropagationError() {
        return propagationError;
    }

    @Override
    public void setLearningData(DoubleMatriz learningData) {
        this.learningData = learningData;
    }

    @Override
    public void startLearning() {
        learningData = (DoubleMatriz)learningData.productoEscalar(1.0/255);
        error = (DoubleMatriz)learningData.substraccion(outputLayer);
//        toBackLayer = (DoubleMatriz) weights.productoPunto(error);
        propagationError = (DoubleMatriz) error.productoPunto(weights).transpuesta();
        
        NumericMatriz<Double> delta = error.productoTensorial(inputLayer).productoEscalar(learningFactor);
        NumericMatriz<Double> adicion = weights.adicion(delta);
        synchronized(weights){
            weights.putAll(adicion);
            
        }
        
        for(LayerLearning back : getProducers()) {
            back.setLearningData(propagationError);
            back.startLearning();
        }
        producers.clear();
        
    }

    @Override
    public DoubleMatriz getError() {
        if( error!=null)
            return (DoubleMatriz)error.distanciaE2().productoEscalar(1.0/2);
        else
            return new DoubleMatriz(new Dominio(1, 1));
    }

    @Override
    public Double getLeanringFactor() {
        return learningFactor;
    }

    @Override
    public List<LayerLearning> getProducers() {
        return producers;
    }

    @Override
    public DoubleMatriz getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {
        
            outputLayer = weights.producto(inputLayer);
            
            for(LayerConsumer lc : consumers) {
                lc.seInputLayer(outputLayer);
                lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
                
                if(lc instanceof LayerLearning) {
                    ((LayerLearning)lc).getProducers().add(this);
                }
            }

    }

    @Override
    public List<LayerConsumer> getConsumers() {
        return consumers;
    }
    
    
    
}
