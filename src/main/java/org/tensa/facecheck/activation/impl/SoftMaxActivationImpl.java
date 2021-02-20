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
package org.tensa.facecheck.activation.impl;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.tensa.facecheck.activation.Activation;
import org.tensa.facecheck.activation.utils.ActivationUtils;
import org.tensa.tensada.matrix.NumericMatriz;

public class SoftMaxActivationImpl<N extends Number> implements Activation<N> {

    @Override
    public Function<NumericMatriz<N>, NumericMatriz<N>> getActivation() {
        return this::getSoftMax;
    }

    private NumericMatriz<N> getSoftMax(NumericMatriz<N> m) {
        Map<Integer, N> sumas = m.entrySet().stream()
                .collect(Collectors.toMap(
                        e -> e.getKey().getColumna(),
                        e -> e.getValue(),
                        (a, b) -> m.sumaDirecta(a, b)));
        return m.entrySet().stream()
                .collect(ActivationUtils.entryToMatriz(
                        m,
                        e -> m.productoDirecto(
                                e.getValue(),
                                m.inversoMultiplicativo(
                                        sumas.get(e.getKey().getColumna())
                                ))
                ));
    }

    @Override
    public BiFunction<NumericMatriz<N>, NumericMatriz<N>, NumericMatriz<N>> getError() {
        return (leraning, output) -> leraning.instancia(output.getDominio());
    }

    @Override
    public boolean isOptimized() {
        return true;
    }

}
