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

import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 */
public final class WeightCreationStyle {

    private WeightCreationStyle() {
    }

    public static <N extends Number> NumericMatriz<N> randomCreationStyle(NumericMatriz<N> tmpm) {
        tmpm.getDominio().forEach((i) -> {
            tmpm.put(i, tmpm.mapper(1-2*Math.random()));
        });
        return tmpm;
    }
    
    public static <N extends Number> NumericMatriz<N> diagonalCreationStyle(NumericMatriz<N> tmpm) {
        Dominio dominio = tmpm.getDominio();
        int pend = dominio.getColumna()/dominio.getFila();
        dominio.stream()
                .filter((p) -> p.getColumna()/p.getFila()== pend)
                .forEach((i) -> {
            tmpm.put(i, tmpm.mapper(1.0));
        });
        return tmpm;
    }
    
    public static <N extends Number> NumericMatriz<N> randomSegmetedCreationStyle(NumericMatriz<N> tmpm) {
        int filas = tmpm.getDominio().getFila() / 3;
        int columnas = tmpm.getDominio().getColumna() / 3;
        tmpm.getDominio().stream()
                .filter((i) -> i.getFila() / filas == i.getColumna() / columnas )
                .forEach((i) -> {
            tmpm.put(i, tmpm.mapper(1-2*Math.random()));
        });
        return tmpm;
    }
    
}
