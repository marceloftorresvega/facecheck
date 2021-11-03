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
package org.tensa.facecheck.element;

/**
 * El elemento de datos esta dividido en unidades, canales (channels) y muestras
 * (samples) estos se pueden colocar (copiar dentro de una posicion) o acceder
 * como Object si se hace la similitud con el pixel cada ElementData en una
 * muestra es un pixel el pixel sin embargo tiene 3 canales R-0,G-1,B-2
 * ElementData tambien puede trabajar a nivel de muestras empaquetadas, es decir
 * un elementData puede empaquetar varios pixels e ir accediendo a cada paquete
 * por muestra
 *
 * la definicion esta abierta a copiar rangos de ElementData a modo de ventanas
 * o mapas
 *
 * @author Marcelo
 */
public interface ElementData {

    /**
     * Obtiene el objeto transferible sample
     *
     * @param sample int ubicacion del sample
     * @return objeto arreglo de elementos
     */
    Object getSample(int sample);

    /**
     * Copia en el element data el elemento transferible sample
     *
     * @param sample int ubicacion del sample
     * @param theSample objeto arreglo de elementos
     */
    void setSample(int sample, Object theSample);

    /**
     * Obtiene el objeto transferible channel para un sample
     *
     * @param channel int ubicacion del channel
     * @param sample int ubicacion del sample
     * @return objeto arreglo de elementos
     */
    Object getChannel(int channel, int sample);

    /**
     * Copia en el element data el objeto transferible channel para un sample
     *
     * @param channel int ubicacion del channel
     * @param sample int ubicacion del sample
     * @param theChannel objeto arreglo de elementos
     */
    void setChannel(int channel, int sample, Object theChannel);

    /**
     * Obtiene el objeto transferible para todos los samples o channels del
     * ElementData
     *
     * @return objeto arreglo de elementos
     */
    Object get();

    /**
     * Copia en el element data el objeto transferible para todos los samples o
     * channels del ElementData
     *
     * @param all objeto arreglo de elementos
     */
    void set(Object all);
}
