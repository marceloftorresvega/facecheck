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

import java.io.IOException;
import java.util.concurrent.RejectedExecutionException;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 */
public class OutputScale {
    private OutputScale() {
    }
    
    public static <N extends Number> NumericMatriz<N> sameEscale(NumericMatriz<N> matriz) {
        return matriz;
    }
    
    public static <N extends Number> NumericMatriz<N> prevent01(NumericMatriz<N> matriz) {
        N escala = matriz.mapper(254.0 / 255.0);
        try (
                NumericMatriz<N> uno = matriz.matrizUno();
                NumericMatriz<N> margen = uno.productoEscalar(matriz.mapper(0.5));
                NumericMatriz<N> escalado = matriz.productoEscalar(escala)) {
            
            return escalado.adicion(margen);
        } catch( IOException ex) {
            throw new RejectedExecutionException("prevent01", ex);
        }
    
    }
    
    public static <N extends Number> NumericMatriz<N> scale(NumericMatriz<N> matriz) {
        N escala = matriz.mapper(1 / 255.0);
        return matriz.productoEscalar(escala);
    }
    
     public static <N extends Number> NumericMatriz<N> normalized(NumericMatriz<N> matriz) {
         try (NumericMatriz<N> d = matriz.distanciaE2();) {
            N normalizador = matriz.inversoMultiplicativo(matriz.mapper(Math.sqrt(d.get(Indice.D1).doubleValue())));
            return matriz.productoEscalar(normalizador);
         
        } catch( IOException ex) {
            throw new RejectedExecutionException("normalized", ex);
        }
            
     }
    
     public static <N extends Number> NumericMatriz<N> reflectance(NumericMatriz<N> matriz) {
         try (NumericMatriz<N> r = matriz.productoPunto(matriz.matrizUno());) {
            
            N reflector = matriz.inversoMultiplicativo(r.get(Indice.D1));
            return matriz.productoEscalar(reflector);
             
        } catch( IOException ex) {
            throw new RejectedExecutionException("reflectance", ex);
        }
     }
     
}
