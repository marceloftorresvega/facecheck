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
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import org.tensa.facecheck.activation.Activation;
import org.tensa.facecheck.activation.utils.ActivationUtils;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * retorna la matriz normal para cada columna de la matriz de ingreso
 *
 * funcion de error de x/ (x2+c): ((x2+c) - 2x2) / (x2 + c)2 -> c-x2 / (x2+c)2
 *
 * @author Marcelo
 * @param <N> clase de nummero utilizado
 */
public class NormalActivationImpl<N extends Number> implements Activation<N> {

    @Override
    public Function<NumericMatriz<N>, NumericMatriz<N>> getActivation() {
        return this::getSoftMax;
    }

    private NumericMatriz<N> getSoftMax(NumericMatriz<N> m) {

        Map<Integer, N> sumas2 = m.entrySet().stream()
                .collect(Collectors.toMap(
                        e -> e.getKey().getColumna(),
                        e -> m.multiplica(e.getValue(), e.getValue()),
                        m::suma));

        return m.entrySet().stream()
                .collect(ActivationUtils.entryToMatriz(
                        m,
                        e -> m.divide(
                                e.getValue(),
                                sumas2.get(e.getKey().getColumna()))
                ));
    }

    @Override
    public BiFunction<NumericMatriz<N>, NumericMatriz<N>, NumericMatriz<N>> getError() {
        return (leraning, neta) -> {

            UnaryOperator<N> cuad = v -> neta.multiplica(v, v);

            NumericMatriz<N> cuadrados = neta.entrySet().stream()
                    .collect(ActivationUtils.entryToMatriz(
                            neta,
                            cuad.compose(Map.Entry::getValue)));

            Map<Integer, N> sumas2 = cuadrados.entrySet().stream()
                    .collect(Collectors.toMap(
                            e -> e.getKey().getColumna(),
                            Map.Entry::getValue,
                            neta::suma));

            return cuadrados.entrySet().stream()
                    .collect(ActivationUtils.entryToMatriz(
                            neta,
                            e -> neta.divide(
                                    neta.resta(
                                            sumas2.get(e.getKey().getColumna()),
                                            neta.multiplica(
                                                    neta.mapper(2.0),
                                                    cuadrados.get(e.getKey()))),
                                    cuad.compose(sumas2::get)
                                            .apply(e.getKey().getColumna())
                            )));
        };
    }

    @Override
    public boolean isOptimized() {
        return false;
    }

}
