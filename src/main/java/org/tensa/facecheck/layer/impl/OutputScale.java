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
import org.tensa.facecheck.activation.utils.CleanMatrix;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 * conjunto de utilidades para escalar una matriz de entrada de pixels
 *
 * @author Marcelo
 */
public final class OutputScale {

    private OutputScale() {
    }

    /**
     * no modifica la matriz de componentes de pixels
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param matriz NumericMatriz Matriz a modificar
     * @return NumericMatriz modificada
     */
    public static <N extends Number> NumericMatriz<N> sameEscale(NumericMatriz<N> matriz) {
        return matriz;
    }

    /**
     * escala la matriz de componentes de pixels para prevenir los valores
     * maximos (255) y minimo 0
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param matriz NumericMatriz Matriz a modificar
     * @return NumericMatriz modificada
     */
    public static <N extends Number> NumericMatriz<N> prevent01(NumericMatriz<N> matriz) {
        N escala = matriz.mapper(254.0 / 255.0);
        try (CleanMatrix<N> s = new CleanMatrix<>()){ 

            return s.ref(matriz.productoEscalar(escala)).adicion(s.ref(s.ref(matriz.matrizUno()).productoEscalar(matriz.mapper(0.5))));
        } catch (IOException ex) {
            throw new RejectedExecutionException("prevent01", ex);
        }

    }

    /**
     * escala la matriz de componentes de pixels entre valores 0 y 1 (preferido)
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param matriz NumericMatriz Matriz a modificar
     * @return NumericMatriz modificada
     */
    public static <N extends Number> NumericMatriz<N> scale(NumericMatriz<N> matriz) {
        N escala = matriz.mapper(1 / 255.0);
        return matriz.productoEscalar(escala);
    }

    /**
     * escala la matriz de componentes de pixels entre valores -1 y 1
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param matriz NumericMatriz Matriz a modificar
     * @return NumericMatriz modificada
     */
    public static <N extends Number> NumericMatriz<N> extendedScale(NumericMatriz<N> matriz) {
        try (CleanMatrix<N> s = new CleanMatrix<>()){ 
            N escala = matriz.mapper(1 / 128.0);
            return s.ref(matriz.productoEscalar(escala)).substraccion(s.ref(matriz.matrizUno()));
        } catch (IOException ex) {
            throw new RejectedExecutionException("extendedScale", ex);
        }
    }

    /**
     * normaliza la matriz de componentes de pixels
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param matriz NumericMatriz Matriz a modificar
     * @return NumericMatriz modificada
     */
    public static <N extends Number> NumericMatriz<N> normalized(NumericMatriz<N> matriz) {
        double punto = matriz.values().stream().mapToDouble(Number::doubleValue).map((double v) -> v * v).sum();
        punto = 1 / Math.sqrt(punto);
        return matriz.productoEscalar(matriz.mapper(punto));
    }

    /**
     * normaliza la matriz de componentes de pixels, asumiendo matriz columna,
     * se reintenta recobrar dimencionalidad perdida
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param matriz NumericMatriz Matriz a modificar
     * @return NumericMatriz modificada
     */
    public static <N extends Number> NumericMatriz<N> normalizedPus1(NumericMatriz<N> matriz) {
        double punto = matriz.values().stream().mapToDouble(Number::doubleValue).map((double v) -> v * v).sum();
        double inversePunto = 1 / Math.sqrt(punto);
        
        try (NumericMatriz<N> scalledMatrix = matriz.productoEscalar(matriz.mapper(inversePunto))){ 
            Integer fila = matriz.getDominio().getFila();
            Integer columna = matriz.getDominio().getColumna();
            Dominio dominio = new Dominio(fila + 1, columna);
            NumericMatriz<N> matrizPlus1 = matriz.instancia(dominio, scalledMatrix);
            matrizPlus1.put((ParOrdenado) dominio, matriz.mapper(inversePunto));

            return matrizPlus1;
        } catch (IOException ex) {
            throw new RejectedExecutionException("normalizedPus1", ex);
        }
    }

    /**
     * normaliza la matriz de componentes de pixels centrandola en intervalo -1
     * y 1
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param matriz NumericMatriz Matriz a modificar
     * @return NumericMatriz modificada
     */
    public static <N extends Number> NumericMatriz<N> extendedNormalized(NumericMatriz<N> matriz) {
        try (CleanMatrix<N> s = new CleanMatrix<>()){ 
            N escala = matriz.mapper(128.0);
            NumericMatriz<N> halfmatriz = s.ref(matriz.substraccion(s.ref(s.ref(matriz.matrizUno()).productoEscalar(escala))));
            double punto = halfmatriz.values().stream().mapToDouble(Number::doubleValue).map((double v) -> v * v).sum();
            punto = 1 / Math.sqrt(punto);
            
            return halfmatriz.productoEscalar(matriz.mapper(punto));
        } catch (IOException ex) {
            throw new RejectedExecutionException("extendedNormalized", ex);
        }
    }

    /**
     * normaliza la matriz de componentes de pixels centrandola en intervalo -1,
     * asumiendo matriz columna, se reintenta recobrar dimencionalidad perdida y
     * 1
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param matriz NumericMatriz Matriz a modificar
     * @return NumericMatriz modificada
     */
    public static <N extends Number> NumericMatriz<N> extendedNormalizedPlus1(NumericMatriz<N> matriz) {
        try (CleanMatrix<N> s = new CleanMatrix<>()){ 
            N escala = matriz.mapper(128.0);
            NumericMatriz<N> halfmatriz = s.ref(matriz.substraccion(s.ref(s.ref(matriz.matrizUno()).productoEscalar(escala))));
            double punto = halfmatriz.values().stream().mapToDouble(Number::doubleValue).map((double v) -> v * v).sum();
            double inversePunto = 1 / Math.sqrt(punto);

            Integer fila = matriz.getDominio().getFila();
            Integer columna = matriz.getDominio().getColumna();
            Dominio dominio = new Dominio(fila + 1, columna);

            NumericMatriz<N> matrizPlus1 = matriz.instancia(dominio, s.ref(halfmatriz.productoEscalar(matriz.mapper(inversePunto))));
            matrizPlus1.put((ParOrdenado) dominio, matriz.mapper(inversePunto));
            return matrizPlus1;
        } catch (IOException ex) {
            throw new RejectedExecutionException("extendedNormalizedPlus1", ex);
        }
    }

    /**
     * aplica operacion de reflectancia a la matriz de componentes de pixels
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param matriz NumericMatriz Matriz a modificar
     * @return NumericMatriz modificada
     */
    public static <N extends Number> NumericMatriz<N> reflectance(NumericMatriz<N> matriz) {
        double punto = matriz.values().stream().mapToDouble((N v) -> Math.abs(v.doubleValue())).sum();
        punto = 1 / punto;
        return matriz.productoEscalar(matriz.mapper(punto));
    }

    /**
     * aplica operacion de reflectancia a la matriz de componentes de pixels a
     * -1 y 1
     *
     * @param <N> tipo numerico Float,Double o BigDecimal
     * @param matriz NumericMatriz Matriz a modificar
     * @return NumericMatriz modificada
     */
    public static <N extends Number> NumericMatriz<N> extendedReflectance(NumericMatriz<N> matriz) {
        try (CleanMatrix<N> s = new CleanMatrix<>()){ 
            N escala = matriz.mapper(128.0);
            NumericMatriz<N> halfmatriz = s.ref(matriz.substraccion(s.ref(s.ref(matriz.matrizUno()).productoEscalar(escala))));
            double punto = halfmatriz.values().stream().mapToDouble((N v) -> Math.abs(v.doubleValue())).sum();
            punto = 1 / punto;
            return halfmatriz.productoEscalar(matriz.mapper(punto));
        } catch (IOException ex) {
            throw new RejectedExecutionException("extendedReflectance", ex);
        }
    }

}
