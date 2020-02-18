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
import java.math.BigDecimal;
import org.tensa.tensada.matrix.BigDecimalMatriz;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 *
 * @author Marcelo
 */
public class BigDecimalLinealLearningLayerImpl extends LearningLayer<BigDecimal> {

    public BigDecimalLinealLearningLayerImpl(NumericMatriz<BigDecimal> weights, BigDecimal learningFactor) {
        super(weights, learningFactor);
    }

    @Override
    public BigDecimal mediaError() {
        return BigDecimal.valueOf(0.5);
    }

    @Override
    public void learningFunctionOperation() {
        error = learningData.substraccion(outputLayer);
//        propagationError = (NumericMatriz<N>) weights.productoPunto(error);
        try ( NumericMatriz<BigDecimal> punto = error.productoPunto(weights)) {

            propagationError =  punto.transpuesta();
        } catch (IOException ex) {
            log.error("learningFunctionOperation", ex);
        }
    }

    @Override
    public BigDecimal activateFunction(ParOrdenado i, BigDecimal value) {
        return value;
    }

    @Override
    protected NumericMatriz<BigDecimal> supplier() {
        return new BigDecimalMatriz(new Dominio(1, 1));
    }
    
}
