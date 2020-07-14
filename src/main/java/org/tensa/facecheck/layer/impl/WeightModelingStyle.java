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
 * Modela estilo de distribucion de pesos generados, considerando para este caso
 * las filas como vectores independientes a tratar se espera en todo caso que
 * cada fila entre con forma de matriz independiente usando una matriz de
 * bloques
 *
 * @author Marcelo
 */
public final class WeightModelingStyle {

    /**
     * normalized: normaliza los vectores
     *
     * @param tmpm NumericMatriz vector a modificar
     */
    public static <N extends Number> NumericMatriz<N> normalizedModelingStyle(NumericMatriz<N> tmpm) {
        double punto = tmpm.values().stream().mapToDouble(Number::doubleValue).map((double v) -> v * v).sum();
        punto = 1 / Math.sqrt(punto);
        return tmpm.productoEscalar(tmpm.mapper(punto));
    }

    /**
     * reflectance: calcula el vector de reflectancia
     *
     * @param tmpm NumericMatriz vector a modificar
     */
    public static <N extends Number> NumericMatriz<N> reflectanceModelingStyle(NumericMatriz<N> tmpm) {
        double punto = tmpm.values().stream().mapToDouble((N v) -> Math.abs(v.doubleValue())).sum();
        punto = 1 / punto;
        return tmpm.productoEscalar(tmpm.mapper(punto));
    }

    /**
     * simple: no afecta al vector
     *
     * @param tmpm NumericMatriz vector a modificar
     */
    public static <N extends Number> NumericMatriz<N> simpleModelingStyle(NumericMatriz<N> tmpm) {
        return tmpm;
    }

    /**
     * Constructor privado
     */
    private WeightModelingStyle() {
    }

}
