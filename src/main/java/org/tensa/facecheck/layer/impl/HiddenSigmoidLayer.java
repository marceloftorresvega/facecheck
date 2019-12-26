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
import java.util.Collections;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.DoubleMatriz;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.facecheck.layer.LayerLearning;

/**
 *
 * @author Marcelo
 */
public class HiddenSigmoidLayer extends ArrayList<LayerConsumer> implements LayerLearning, LayerConsumer, LayerProducer {
    
    private final Logger log = LoggerFactory.getLogger(HiddenSigmoidLayer.class);
    
    private final DoubleMatriz weights;
    private int status;
    private DoubleMatriz outputLayer;
    private DoubleMatriz inputLayer;
    
    private DoubleMatriz propagationError;
    private DoubleMatriz learningData;
    private DoubleMatriz error;
    private final Double learningFactor;

    public HiddenSigmoidLayer(DoubleMatriz weights, Double learningStep) {
        this.weights = weights;
        this.learningFactor = learningStep;
    }

    @Override
    public DoubleMatriz seInputLayer(DoubleMatriz inputLayer) {
        this.inputLayer = inputLayer;
        return null;
    }

    @Override
    public DoubleMatriz getOutputLayer() {
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
            
//            DoubleMatriz producto = weights.producto(inputLayer);
//            DoubleMatriz distanciaE2 = (DoubleMatriz)producto.distanciaE2();
//            outputLayer = (DoubleMatriz)producto
//                    .productoEscalar( 1 / Math.sqrt(distanciaE2.get(Indice.D1)));
            outputLayer = weights.producto(inputLayer);
            outputLayer.replaceAll((i,v) -> 1/(1 + Math.exp( - v )));
            
            for(LayerConsumer lc : this) {
                lc.seInputLayer(outputLayer);
                lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
                
                if(lc instanceof LayerLearning) {
                    ((LayerLearning)lc).getProducers().add(this);
                }
            }
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
        error = (DoubleMatriz) outputLayer.matrizUno().substraccion(outputLayer);
        error.replaceAll((i,v) -> v * outputLayer.get(i) * learningData.get(i));
        
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
        getProducers().clear();
    }

    @Override
    public DoubleMatriz getWeights() {
        return weights;
    }

    @Override
    public DoubleMatriz getError() {
        return (DoubleMatriz)error.distanciaE2().productoEscalar(1.0/2);
    }

    @Override
    public Double getLeanringFactor() {
        return learningFactor;
    }

    @Override
    public List<LayerLearning> getProducers() {
        return Collections.emptyList();
    }

    @Override
    public List<LayerConsumer> getConsumers() {
       return this;
    }
    
}
