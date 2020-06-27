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
import org.tensa.tensada.matrix.Indice;

/**
 * mapeo de componentes desde el arreglo de pixels
 * 1.- directo al arreglo de componentes de pixels 
 * 2.- separacion en columnas 1a rojo 2a verde 3a azul 
 * 3.- separacion en columnas y alineacion 1a rojo 2a 
 *     verde 3a azul y superior rojo,
 *     centro verde e inferior azul
 *
 * @author Marcelo
 */
public final class PixelMappings {

    private PixelMappings() {
    }

    /**
     * mapeo directo de los pixels separados en componentes
     *
     * @return PixelMapper
     */
    public static PixelMapper defaultMapping() {
        return new PixelMapper() {
            @Override
            public Dominio getDominio(int largo) {
                return new Dominio(largo, 1);
            }

            @Override
            public Indice getIndice(int i) {
                return new Indice(i + 1, 1);
            }

            @Override
            public Integer getLargo(Dominio dominio) {
                return dominio.getFila();
            }
        };
    }

    /**
     * mapeo y clasificacion de componentes de pixels en 3 columnas, reduciendo
     * cantidad de filas con respecto a la cantidad total de componentes
     *
     * @return PixelMapper
     */
    public static PixelMapper byComponentMapping() {
        return new PixelMapper() {
            @Override
            public Dominio getDominio(int largo) {
                return new Dominio(largo / 3, 3);
            }

            @Override
            public Indice getIndice(int i) {
                return new Indice(i / 3 + 1, i % 3 + 1);
            }

            @Override
            public Integer getLargo(Dominio dominio) {
                return dominio.getFila() * 3;
            }
        };
    }

    /**
     * mapeo de componentes del pixel en columnas y recolocacion de las
     * componentes dentro de la fila clasificandolas por color, conserva largo
     * de la cantidad de componentes
     *
     * @return PixelMapper
     */
    public static PixelMapper bySegmentedComponentMapping() {
        return new PixelMapper() {
            private Dominio dominioLocal;

            @Override
            public Dominio getDominio(int largo) {
                return this.dominioLocal = new Dominio(largo, 3);
            }

            @Override
            public Indice getIndice(int i) {
                return new Indice((i % 3) * dominioLocal.getFila() + i / 3 + 1, i % 3 + 1);
            }

            @Override
            public Integer getLargo(Dominio dominio) {
                this.dominioLocal = dominio;
                return dominio.getFila();
            }
        };
    }

    /**
     * mapeo inverso a byComponentMapping donde se distribullen en 3 filas y el
     * largo de la columna sea reducido con respecto a la cantidad de
     * componentes
     *
     * @return PixelMapper
     */
    public static PixelMapper bySampleMapping() {
        return new PixelMapper() {
            @Override
            public Dominio getDominio(int largo) {
                return new Dominio(3, largo / 3);
            }

            @Override
            public Indice getIndice(int i) {
                return new Indice(i % 3 + 1, i / 3 + 1);
            }

            @Override
            public Integer getLargo(Dominio dominio) {
                return dominio.getColumna() * 3;
            }
        };
    }
}
