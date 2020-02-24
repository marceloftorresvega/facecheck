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

import java.util.function.BiFunction;
import java.util.function.Function;
import org.tensa.tensada.matrix.NumericMatriz;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 *
 * @author Marcelo
 */
public class LinealActivationImpl<N extends Number> implements org.tensa.facecheck.activation.Activation<N> {
    private final Function<Double, N> mapper;

    public LinealActivationImpl(Function<Double, N> mapper) {
        this.mapper = mapper;
    }


    @Override
    public Function<NumericMatriz<N>, NumericMatriz<N>> getActivation() {
        return Function.identity();
    }

    @Override
    public BiFunction<NumericMatriz<N>, NumericMatriz<N>, NumericMatriz<N>> getError() {
        return (learning,output) -> learning.substraccion(output);
    }

    @Override
    public Function<Double, N> getMapper() {
        return mapper;
    }

}
