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
 * Mapeador de pixeles 
 * genera dominio para matrices y obtiene largo de datasets de pixels
 * mapea posicion de dataset a posicion dentro de la matriz
 * @author Marcelo
 */
public interface PixelMapper {
    
    /**
     * @param length the value of length
     * @param slot the value of slot
     */
    Dominio getDominioBuffer(int length, int slot);
    
    /**
     * genera dominio columna en base a largo d dataset
     * @param largo
     * @return dominio para matriz
     */
    Dominio getDominio(int largo);
    
    /**
     * mapea un pixel o componente en base a su posicion en el dataset
     * @param i posicion en el dataset
     * @return indice para la matriz
     */
    Indice getIndice(int i);
    
    /**
     * mapea un pixel o componente en base a su posicion en el dataset y el slot correspondiente
     * @param i
     * @param slot
     * @return indice para la matriz
     */
    Indice getIndice(int i, int slot);
    
    /**
     * retorna el largo de un data set en base al dominio de la matriz
     * @param dominio
     * @return largo de dataset
     */
    Integer getLargo(Dominio dominio);
    
}
