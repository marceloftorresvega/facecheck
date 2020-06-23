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

import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 *
 * @author Marcelo
 */
public class ActivationUtils {

    private ActivationUtils() {
    }
    
    
    public static <N extends Number> Collector<ParOrdenado, ?, NumericMatriz<N>> domainToMatriz(NumericMatriz<N> m, Function<ParOrdenado,N> convert){
        return Collectors.toMap(
                Function.identity(),
                convert,
                (a,b) -> a,
                () -> m.instancia(m.getDominio()));
    }
    
    public static <N extends Number> Collector<? super Map.Entry<ParOrdenado, N>, ?, NumericMatriz<N>> entityToMatriz(NumericMatriz<N> m, Function<Map.Entry<ParOrdenado, N>,N> convert){
        return Collectors.toMap(
                NumericMatriz.Entry::getKey,
                convert,
                (a,b) -> a,
                () -> m.instancia(m.getDominio()));
    }
}
