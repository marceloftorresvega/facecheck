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
 * Modela creacion de pesos para las matrices al ser iniciadas
 *
 * @author Marcelo
 */
public final class WeightCreationStyle {

    private WeightCreationStyle() {
    }

    /**
     * random: asigna a los pesos valores al azar entre -1 y 1
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param tmpm NumericMatriz Matriz a rellenar
     * @return NumericMatriz la matriz ingresada
     */
    public static <N extends Number> NumericMatriz<N> randomCreationStyle(NumericMatriz<N> tmpm) {
        tmpm.getDominio().forEach((i) -> {
            tmpm.put(i, tmpm.mapper(1 - 2 * Math.random()));
        });
        return tmpm;
    }

    /**
     * diagonal: asigna a los pesos valores 1 si se situan sobre la diagonal, no
     * requiere matriz cuadarada
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param tmpm NumericMatriz Matriz a rellenar
     * @return NumericMatriz la matriz ingresada
     */
    public static <N extends Number> NumericMatriz<N> diagonalCreationStyle(NumericMatriz<N> tmpm) {
        Dominio dominio = tmpm.getDominio();
        int pend = dominio.getColumna() / dominio.getFila();
        dominio.stream()
                .filter((p) -> p.getColumna() / p.getFila() == pend)
                .forEach((i) -> {
                    tmpm.put(i, tmpm.mapper(1.0));
                });
        return tmpm;
    }
    
    public static <N extends Number> NumericMatriz<N> fromOutStarCreationStyle(NumericMatriz<N> tmpm) {
        Dominio dominio = tmpm.getDominio();
        int pend = dominio.getColumna()/dominio.getFila() / 2;
        int limit = dominio.getFila() / 2;
        dominio.stream()
                .filter((p) -> p.getColumna()/p.getFila()== pend || p.getColumna()/(p.getFila() - limit )== pend )
                .forEach((i) -> {
                    if (i.getFila()< limit) {
                        tmpm.put(i, tmpm.mapper(1.0));
                        
                    } else {
                        tmpm.put(i, tmpm.mapper(-1.0));
                        
                    }
        });
        return tmpm;
    }
    

    /**
     * segmented: inicia matriz para modelo pixelMapping que divide componentes
     * de pixel en grupos
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param tmpm NumericMatriz Matriz a rellenar
     * @return NumericMatriz la matriz ingresada
     */
    public static <N extends Number> NumericMatriz<N> randomSegmetedCreationStyle(NumericMatriz<N> tmpm) {
        int filas = tmpm.getDominio().getFila() / 3;
        int columnas = tmpm.getDominio().getColumna() / 3;
        tmpm.getDominio().stream()
                .filter((i) -> i.getFila() / filas == i.getColumna() / columnas)
                .forEach((i) -> {
                    tmpm.put(i, tmpm.mapper(1 - 2 * Math.random()));
                });
        return tmpm;
    }

}
