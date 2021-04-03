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

import java.util.function.BiFunction;
import java.util.function.Function;
import org.tensa.facecheck.activation.Activation;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 * @param <N>
 */
public class ComposeActivationImpl<N extends Number> implements Activation<N> {

    private final Activation<N> inicial;
    private final Activation<N> ultima;

    public ComposeActivationImpl(Activation<N> inicial, Activation<N> ultima) {
        if(!inicial.isOptimized()) {
            throw new IllegalArgumentException("parametro inicial debe ser optimizado.");
        }
        this.inicial = inicial;
        this.ultima = ultima;
    }
    
    @Override
    public Function<NumericMatriz<N>, NumericMatriz<N>> getActivation() {
        return ultima.getActivation().compose(inicial.getActivation());
    }

    @Override
    public BiFunction<NumericMatriz<N>, NumericMatriz<N>, NumericMatriz<N>> getError() {
        return (learning, output) -> ultima.getError().andThen(m -> inicial.getError().apply(learning.matrizUno(), m)).apply(learning, output);
    }

    @Override
    public boolean isOptimized() {
        return ultima.isOptimized();
    }
    
}
