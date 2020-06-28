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
package org.tensa.facecheck.activation;

import java.util.function.BiFunction;
import java.util.function.Function;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * Interface que provee funcion de activacion para las neuronas de una capa,
 * ademas provee funcion de error interno e indica si esta ultima es optimizada
 *
 * @author Marcelo
 */
public interface Activation<N extends Number> {

    /**
     * provee funcion de activacion para la capa, donde la matriz de entrada es
     * la salida neta de las neuronas de la capa
     *
     * @return Function matriz neta a matriz salida
     */
    Function<NumericMatriz<N>, NumericMatriz<N>> getActivation();

    /**
     * provee funcion que aplica error externo a derivada de la funcion de
     * activacion la porpiedad optimized inica si usa el valor output o si no el
     * valor neto antes de la activacion
     *
     * @return Function matriz error externo y output capa a matriz de error
     * interno
     */
    BiFunction<NumericMatriz<N>, NumericMatriz<N>, NumericMatriz<N>> getError();

    /**
     * indica si la funcion de calculo de error interno recibe el valor
     * pre-calculado desde el output de la capa o si debe procesar la salida
     * neta de las neuronas de la capa
     *
     * @return booleano output(true)/neta(false)
     */
    boolean isOptimized();
}
