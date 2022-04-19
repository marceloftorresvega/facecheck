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
package org.tensa.facecheck.activation.utils;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 * limpia conjunto de matrices
 * @author Marcelo
 */
public class CleanMatrix<N extends Number> implements Closeable {
    
    private final ArrayList<NumericMatriz<N>> list;

    public CleanMatrix() {
        this.list = new ArrayList<>();
    }

    @Override
    public void close() throws IOException {
        for( Closeable element : list) {
            element.close();
        }
        list.clear();
    }
    
    public NumericMatriz<N> ref(NumericMatriz<N> matrix) {
        list.add(matrix);
        return matrix;
    }
}
