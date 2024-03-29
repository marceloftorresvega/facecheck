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
package org.tensa.facecheck.activation.impl;

import org.tensa.facecheck.activation.utils.ActivationUtils;
import java.util.function.BiFunction;
import java.util.function.Function;
import org.tensa.facecheck.activation.Activation;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * Implementacion funcion de activacion para regresion lineal y uso de derivada
 * para calculo de error, optimizada para el uso de la salida de la capa
 *
 * @author Marcelo
 * @param <N>
 */
public class ReluActivationImpl<N extends Number> implements Activation<N> {

    @Override
    public Function<NumericMatriz<N>, NumericMatriz<N>> getActivation() {
        return (m) -> {
            return m.entrySet().stream()
                    .filter((e) -> e.getValue().doubleValue() > 0.0)
                    .collect(ActivationUtils.entryToMatriz(m,
                            NumericMatriz.Entry::getValue));
        };
    }

    @Override
    public BiFunction<NumericMatriz<N>, NumericMatriz<N>, NumericMatriz<N>> getError() {
        return (NumericMatriz<N> learning, NumericMatriz<N> output) -> {
//            NumericMatriz<N> derivate = output.entrySet().stream()
//                    .filter((e) -> e.getValue().doubleValue() > 0.0)
//                    .collect(toMap(
//                            NumericMatriz.Entry::getKey,
//                            (e) -> output.getUnoValue(),
//                            (a,b) -> a,
//                            () -> output.instancia(output.getDominio())));

            return output.entrySet().stream()
                    .filter((e) -> e.getValue().doubleValue() > 0.0)
                    .collect(ActivationUtils.entryToMatriz(output,
                            (e) -> output.resta(
                                    learning.get(e.getKey()),
                                    e.getValue())));
        };
    }

    @Override
    public boolean isOptimized() {
        return true;
    }

}
