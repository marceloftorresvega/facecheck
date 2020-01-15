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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.DoubleMatriz;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 *
 * @author lorenzo
 */
public class HiddenSimpleCohonenLayer implements LayerLearning, LayerConsumer, LayerProducer {

    private DoubleMatriz inputLayer;
    private final DoubleMatriz weights;
    private DoubleMatriz outputLayer;
    private final List<LayerConsumer> consumer;
    private final List<LayerLearning> producers;
    private int status;
    private Double learningFactor;
    private DoubleMatriz propagationError;

    public HiddenSimpleCohonenLayer(DoubleMatriz weights, Double learningFactor) {
        this.weights = weights;
        this.learningFactor = learningFactor;
        this.consumer = new ArrayList<>();
        this.producers = new ArrayList<>();
    }

    @Override
    public DoubleMatriz getPropagationError() {
        return this.propagationError;
    }

    @Override
    public DoubleMatriz getError() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Double getLeanringFactor() {
        return this.learningFactor;
    }

    @Override
    public void setLearningData(DoubleMatriz learningData) {
        
    }

    @Override
    public void startLearning() {
        propagationError = (DoubleMatriz) inputLayer.substraccion(
                outputLayer.productoPunto(weights).transpuesta());
        
        NumericMatriz<Double> delta = outputLayer.productoTensorial(
                propagationError).productoEscalar(learningFactor);
        
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
    public List<LayerLearning> getProducers() {
        return producers;
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
        this.startProduction();
    }

    @Override
    public DoubleMatriz getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {

        if (status == LayerConsumer.SUCCESS_STATUS) {
            //en caso de tener pesos normalizados se puede usar el producto como expresion de diferencia
            //DoubleMatriz valorNeto = weights.producto(inputLayer);

            //en caso de pesos sin normalizar se emplea esta expresion
            NumericMatriz<Double> matrizUno = weights.matrizUno(new Dominio(weights.getDominio().getFila(),1));
            NumericMatriz<Double> substraccion = matrizUno.productoTensorial(inputLayer).substraccion(weights);
            
            NumericMatriz<Double> distancias2 = substraccion.productoTensorial(substraccion);
            distancias2.replaceAll((i,v) -> i.getFila()==i.getColumna()?v:0.0);
            NumericMatriz<Double> valorNeto = distancias2.producto(matrizUno);
            
            Optional<Map.Entry<ParOrdenado, Double>> max = valorNeto.entrySet().stream().max((e1, e2) -> e1.getValue().compareTo(e2.getValue()));

            outputLayer = new DoubleMatriz(valorNeto.getDominio());
            if (max.isPresent()) {
                outputLayer.put(max.get().getKey(), 1.0);
            }

            for (LayerConsumer lc : consumer) {
                lc.seInputLayer(outputLayer);
                lc.layerComplete(LayerConsumer.SUCCESS_STATUS);

                if (lc instanceof LayerLearning) {
                    ((LayerLearning) lc).getProducers().add(this);
                }
            }
        }
    }

    @Override
    public List<LayerConsumer> getConsumers() {
        return consumer;
    }

}
