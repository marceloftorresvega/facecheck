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
import java.io.IOException;
import java.util.function.BiFunction;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.activation.Activation;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * Implementacion de funcion de activacion tangente hiperbolica y su derivada
 * para calculo de error, optimizada para el uso de la salida de la capa
 *
 * @author Marcelo
 * @param <N>
 */
public class TanHyperActivationImpl<N extends Number> implements Activation<N> {

    protected final Logger log = LoggerFactory.getLogger(TanHyperActivationImpl.class);

    @Override
    public Function<NumericMatriz<N>, NumericMatriz<N>> getActivation() {
        return (m) -> m.entrySet().stream()
                .collect(ActivationUtils.entryToMatriz(
                        m,
                        (e) -> m.tanh(e.getValue())));
    }

    @Override
    public BiFunction<NumericMatriz<N>, NumericMatriz<N>, NumericMatriz<N>> getError() {
        return (NumericMatriz<N> learning, NumericMatriz<N> output) -> {

            try (final NumericMatriz<N> m1 = output.matrizUno();
                    final NumericMatriz<N> semiResta = m1.substraccion(output);
                    final NumericMatriz<N> semiSuma = m1.adicion(output);) {

                return learning.entrySet().stream()
                        .collect(ActivationUtils.entryToMatriz(
                                learning,
                                (e) -> m1.multiplica(
                                        e.getValue(),
                                        m1.multiplica(
                                                semiSuma.get(e.getKey()),
                                                semiResta.get(e.getKey())))));

            } catch (IOException ex) {
                log.error("learningFunctionOperation", ex);
                throw new RuntimeException(ex);
            }

        };
    }

    @Override
    public boolean isOptimized() {
        return true;
    }

}
