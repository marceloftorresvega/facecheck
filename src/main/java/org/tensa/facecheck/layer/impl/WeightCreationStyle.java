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

import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 */
public class WeightCreationStyle {

    private WeightCreationStyle() {
    }

    /**
     *
     * @param tmpm the value of tmpm
     */
    public static <N extends Number> NumericMatriz<N> normalCreationStyle(NumericMatriz<N> tmpm) {
        double punto = tmpm.values().stream().mapToDouble(Number::doubleValue).map((double v) -> v * v).sum();
        punto = 1 / Math.sqrt(punto);
        return tmpm.productoEscalar(tmpm.mapper(punto));
    }

    /**
     *
     * @param tmpm the value of tmpm
     */
    public static <N extends Number> NumericMatriz<N> reflectCreationStyle(NumericMatriz<N> tmpm) {
        double punto = tmpm.values().stream().mapToDouble((N v) -> Math.abs(v.doubleValue())).sum();
        punto = 1 / punto;
        return tmpm.productoEscalar(tmpm.mapper(punto));
    }

    /**
     *
     * @param tmpm the value of tmpm
     */
    public static <N extends Number> NumericMatriz<N> simpleCreationStyle(NumericMatriz<N> tmpm) {
        return tmpm;
    }
    
}
