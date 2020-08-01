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
package org.tensa.facecheck.mapping;

import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 * conjunto de IndexMappers
 *
 * @author Marcelo
 */
public class IndexMappings {

    private IndexMappings() {
    }

    /**
     * mapea indice a valores escalados con respecto a su maximo dominio
     *
     * @param <N>
     * @param dom dominio de entrada del indice
     * @return funcional IndexMapper
     */
    public static <N extends Number> IndexMapper<N> getScaledMapper(Dominio dom) {
        return new IndexMapper<N>() {
            @Override
            public NumericMatriz<N> apply(ParOrdenado idx, NumericMatriz<N> matriz) {
                double filas = dom.getFila().doubleValue();
                double columnas = dom.getColumna().doubleValue();
                double fila = idx.getFila().doubleValue();
                double columna = idx.getColumna().doubleValue();

                matriz.indexa(1, 1, matriz.mapper(fila / filas));
                matriz.indexa(2, 1, matriz.mapper(columna / columnas));
                return matriz;
            }

            @Override
            public Dominio getDominio() {
                return new Dominio(2, 1);
            }
        };
    }

    /**
     * genera mapeo del indice entre un total de entradas equivalente al total
     * del dominio
     *
     * @param <N>
     * @param dom dominio de entrada del indice
     * @return funcional IndexMapper
     */
    public static <N extends Number> IndexMapper<N> getExtremeIdentityMapper(Dominio dom) {
        return new IndexMapper<N>() {
            @Override
            public Dominio getDominio() {
                int filas = dom.getFila();
                int columnas = dom.getColumna();

                return new Dominio(filas * columnas, 1);
            }

            @Override
            public NumericMatriz<N> apply(ParOrdenado idx, NumericMatriz<N> matriz) {
                int filas = dom.getFila();
                int fila = idx.getFila();
                int columna = idx.getColumna();

                int posicion = (fila - 1 + filas * (columna - 1)) + 1;

                matriz.indexa(posicion, 1, matriz.getUnoValue());
                return matriz;
            }
        };
    }

    /**
     * genera un mapeo que asigna entradas para el valor en fila y columna
     *
     * @param <N>
     * @param dom dominio de entrada del indice
     * @return funcional IndexMapper
     */
    public static <N extends Number> IndexMapper<N> getIdentityMapper(Dominio dom) {
        return new IndexMapper<N>() {
            @Override
            public Dominio getDominio() {
                int filas = dom.getFila();
                int columnas = dom.getColumna();
                return new Dominio(filas + columnas, 1);
            }

            @Override
            public NumericMatriz<N> apply(ParOrdenado idx, NumericMatriz<N> matriz) {
                int filas = dom.getFila();
                int fila = idx.getFila();
                int columna = idx.getColumna();

                matriz.indexa(fila, 1, matriz.getUnoValue());
                matriz.indexa(filas + columna, 1, matriz.getUnoValue());
                return matriz;
            }
        };
    }
}
