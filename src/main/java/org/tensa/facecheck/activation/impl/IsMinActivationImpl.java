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

import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.tensa.facecheck.activation.Activation;
import static org.tensa.facecheck.activation.utils.ActivationUtils.entryToMatriz;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * retorna matriz de resultados basados en la deteccion del valor mas bajo
 * dentro de cada columna
 *
 * @author Marcelo
 * @param <N> clase de nummero utilizado
 */
public class IsMinActivationImpl<N extends Number> implements Activation<N> {

    @Override
    public Function<NumericMatriz<N>, NumericMatriz<N>> getActivation() {
        return this::getIsMin;
    }

    private NumericMatriz<N> getIsMin(NumericMatriz<N> m) {
        return m.entrySet().stream()
                .collect(Collectors.groupingBy(
                        e -> e.getKey().getColumna(),
                        Collectors.minBy((e1, e2) -> Double.compare(e1.getValue().doubleValue(), e2.getValue().doubleValue()))
                )).values().stream()
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(entryToMatriz(m, e -> m.getUnoValue()));
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
