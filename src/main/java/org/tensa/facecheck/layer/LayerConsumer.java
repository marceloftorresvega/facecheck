/*
 * The MIT License
 *
 * Copyright 2019 Marcelo.
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
package org.tensa.facecheck.layer;

import org.tensa.tensada.matrix.NumericMatriz;

/**
 * consumidor de resultados de otra capa
 * @author Marcelo
 * @param <N>
 */
public interface LayerConsumer<N extends Number> {
    
    /**
     * estado de proceso correcto
     */
    public static final int SUCCESS_STATUS = 0;

    /**
     * estado de proceso erroneo
     */
    public static final int ERROR_STATUS = 1;
    
    /**
     * carga datos de entrada producido por otra capa
     * @param inputLayer
     * @return the org.tensa.tensada.matrix.NumericMatriz<N>
     */
    NumericMatriz<N> setInputLayer(org.tensa.tensada.matrix.NumericMatriz<N> inputLayer);
    
    /**
     * consume datos de entrada y realiza proceso por los pesos
     * @param status @see LayerConsumer.SUCCESS_STATUS suministrado al procedimiento
     */
    void layerComplete(int status);
    
}
