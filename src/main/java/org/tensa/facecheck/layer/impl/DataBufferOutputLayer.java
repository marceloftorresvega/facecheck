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
package org.tensa.facecheck.layer.impl;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.math.BigDecimal;
import java.util.function.BiConsumer;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 * @param <N>
 */
public class DataBufferOutputLayer<N extends Number> implements LayerConsumer<N> {

    private final OutputStream bos;
    private final BiConsumer<ObjectOutputStream, N> writed;
    private NumericMatriz<N> inputLayer;

    public DataBufferOutputLayer(OutputStream bos, BiConsumer<ObjectOutputStream, N> writed) {
        this.bos = bos;
        this.writed = writed;
    }

    @Override
    public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
        NumericMatriz<N> last = this.inputLayer;
        this.inputLayer = inputLayer;
        return last;
    }

    @Override
    public void layerComplete(int status) {
        if(LayerConsumer.ERROR_STATUS == status) {
            return;
        }

        try (final ObjectOutputStream oos = new ObjectOutputStream(bos)) {
            this.inputLayer.getDominio().forEach(i -> writed.accept(oos, this.inputLayer.get(i)));
        } catch (IOException | UncheckedIOException ex) {
            //
        }
    }

    public void floatWrite(ObjectOutputStream os, N n) {
        try {
            os.writeFloat(n.floatValue());
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

    public void doubleWrite(ObjectOutputStream os, N n) {
        try {
            os.writeDouble(n.doubleValue());
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

    public void bigDecimalWrite(ObjectOutputStream os, N n) {
        try {
            os.writeObject(((BigDecimal) n));
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

}
