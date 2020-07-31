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
import java.util.function.BooleanSupplier;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * capa que parmite el retorno de la señal de error si y solo si se cumple la
 * condicion, la negacion esta incluida
 *
 * @author Marcelo
 * @param <N>
 */
public class BackDoorLayer<N extends Number> implements LayerLearning<N> {

    /**
     * inicia capa de retorno con su funcion de evaluacion
     *
     * @param condition
     */
    public BackDoorLayer(BooleanSupplier condition) {
        this.condition = condition;
    }

    private final BooleanSupplier condition;
    private NumericMatriz<N> learningData;
    private NumericMatriz<N> propagationError;
    private NumericMatriz<N> elsePropagationError;
    private List<LayerLearning<N>> producers = new ArrayList<>();
    private List<LayerLearning<N>> elseProducers = new ArrayList<>();
    private final LayerLearning<N> elseLearning = new LayerLearning<N>() {
        @Override
        public NumericMatriz<N> getPropagationError() {
            return elsePropagationError;
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
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public void startLearning() {
            elsePropagationError = learningData;
            for (LayerLearning<N> back : getProducers()) {
                back.setLearningData(elsePropagationError);
                back.startLearning();
            }
        }

        @Override
        public List<LayerLearning<N>> getProducers() {
            return elseProducers;
        }
    };

    @Override
    public NumericMatriz<N> getPropagationError() {
        return this.propagationError;
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
        this.learningData = learningData;
    }

    @Override
    public void startLearning() {
        if (condition.getAsBoolean()) {
            propagationError = learningData;
            for (LayerLearning<N> back : getProducers()) {
                back.setLearningData(propagationError);
                back.startLearning();
            }
        } else {
            this.elseLearning.startLearning();
        }
    }

    @Override
    public List<LayerLearning<N>> getProducers() {
        return this.producers;
    }

    /**
     * learning que conduce datos en caso de no cumplir la condicion
     *
     * @return LayerLearning
     */
    public LayerLearning<N> getElseLearning() {
        return elseLearning;
    }

}
