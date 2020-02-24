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

import java.util.List;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * ejecucion de metodos backpropagation
 * calcula valores de error y propaga
 * @author Marcelo
 * @param <N>
 */
public interface LayerLearning<N extends Number> {
    
    /**
     * resultados para la propagacion del error
     * @return the org.tensa.tensada.matrix.DoubleMatriz
     */
    NumericMatriz<N> getPropagationError();
    
    /**
     * error medio de la capa
     * @return matriz de error (1 valor)
     */
    NumericMatriz<N> getError();
    
    /**
     * retorna valor de ajuste para los pesos
     * @return the java.lang.Double
     */
    N getLeanringFactor();
    
    /**
     * se adquiere valores deseados para aprendisaje supervizado
     * @param learningData matriz de valores para comparar
     */
    void setLearningData(NumericMatriz<N> learningData);
    
    /**
     * procesa valores deseados versus valores producidos y propaga error
     */
    void startLearning();
    
    /**
     * retorna el listado de capas que reciben la propagacion del error
     * @return listado de capas
     */
    List<LayerLearning<N>> getProducers();
    
    /**
     * funcion de aprendisaje
     */
    void calculateErrorOperation();
}
