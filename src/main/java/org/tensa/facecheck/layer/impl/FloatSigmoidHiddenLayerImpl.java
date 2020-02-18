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

import java.io.IOException;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 *
 * @author Marcelo
 */
public class FloatSigmoidHiddenLayerImpl extends HiddenLayer<Float> {

    public FloatSigmoidHiddenLayerImpl(NumericMatriz<Float> weights, Float learningFactor) {
        super(weights, learningFactor);
    }

    @Override
    public void learningFunctionOperation() {
            try (final NumericMatriz<Float> m1 = outputLayer.matrizUno()) {
                error = (NumericMatriz<Float>) m1.substraccion(outputLayer);
                error.replaceAll((ParOrdenado i, Float v) -> v * outputLayer.get(i) * learningData.get(i));
            } catch (IOException ex) {
                log.error("learningFunctionOperation", ex);
            }
    }

    @Override
    public Float activateFunction(ParOrdenado i, Float value) {
        return (float) (1 / (1 + Math.exp(-value.doubleValue())));
    }

    @Override
    public Float mediaError() {
        return 0.5f;
    }

}
