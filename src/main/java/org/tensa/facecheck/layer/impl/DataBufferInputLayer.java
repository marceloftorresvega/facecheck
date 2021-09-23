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

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.UncheckedIOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 * @param <N>
 */
public class DataBufferInputLayer<N extends Number> implements LayerProducer<N> {

    private final InputStream inputStream;
    private final Function<ObjectInputStream, N> readed;
    private NumericMatriz<N> outputLayer;
    private final List<LayerConsumer<N>> consumers;
    private final Supplier<NumericMatriz<N>> supplier;

    public DataBufferInputLayer(InputStream inputStream, Function<ObjectInputStream, N> readed, Supplier<NumericMatriz<N>> supplier) {
        this.inputStream = inputStream;
        this.readed = readed;
        this.consumers = new ArrayList<>();
        this.supplier = supplier;
    }

    @Override
    public NumericMatriz<N> getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {
        try (final ObjectInputStream ois = new ObjectInputStream(inputStream)) {
            outputLayer = supplier.get();
            outputLayer.getDominio().forEach(i -> outputLayer.put(i, readed.apply(ois)));

            for (LayerConsumer<N> lc : consumers) {
                lc.setInputLayer(outputLayer);
            }

            for (LayerConsumer<N> lc : consumers) {
                lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
            }
        } catch (IOException | UncheckedIOException ex) {
            
//            for (LayerConsumer<N> lc : consumers) {
//                lc.layerComplete(LayerConsumer.ERROR_STATUS);
//            }
        }
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return consumers;
    }

    public N floatRead(ObjectInputStream is) {
        try {
            return (N) Float.valueOf(is.readFloat());
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

    public N doubleRead(ObjectInputStream is) {
        try {
            return (N) Double.valueOf(is.readDouble());
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

    @Deprecated
    public N bigDecimalRead(ObjectInputStream is) {
        try {
            return (N) (BigDecimal) is.readObject();
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        } catch (ClassNotFoundException ex) {
            throw new RuntimeException(ex);
        }
    }

}
