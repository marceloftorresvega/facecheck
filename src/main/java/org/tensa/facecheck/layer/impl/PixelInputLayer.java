/*
 * The MIT License
 *
 * Copyright 2020 lorenzo.
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

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.facecheck.mapping.PixelMapper;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author Marcelo
 * @param <N>
 */
public class PixelInputLayer<N extends Number> implements LayerProducer<N> {

    protected BufferedImage src;
    protected NumericMatriz<N> outputLayer;
    private final List<BufferedImage> srcList;
    protected final List<LayerConsumer<N>> consumers;
    protected final Function<Dominio, NumericMatriz<N>> supplier;
    protected final UnaryOperator<NumericMatriz<N>> responceEscale;
    protected final PixelMapper pixelMapper;
    private final int slotBuffer;

    public PixelInputLayer(BufferedImage src, Function<Dominio, NumericMatriz<N>> supplier, UnaryOperator<NumericMatriz<N>> responceEscale, PixelMapper pixelMapper, int slotBuffer) {
        this.src = src;
        this.consumers = new ArrayList<>();
        this.srcList = new ArrayList<>();
        this.srcList.add(src);
        this.supplier = supplier;
        this.responceEscale = responceEscale;
        this.pixelMapper = pixelMapper;
        this.slotBuffer = slotBuffer;
    }

    public PixelInputLayer(Function<Dominio, NumericMatriz<N>> supplier, UnaryOperator<NumericMatriz<N>> responceEscale, PixelMapper pixelMapper, int slotBuffer) {
        this.src = null;
        this.srcList = new ArrayList<>();
        this.consumers = new ArrayList<>();
        this.supplier = supplier;
        this.responceEscale = responceEscale;
        this.pixelMapper = pixelMapper;
        this.slotBuffer = slotBuffer;
    }

    protected NumericMatriz<N> scanInput() {
        if (src == null && srcList.isEmpty()) {
            throw new NullPointerException("src image is null");
        }
            
        if ( src == null) {
            src = srcList.get(0);
        }

        int width = src.getWidth();
        int height = src.getHeight();

        NumericMatriz<N> dm = supplier.compose(size -> pixelMapper.getDominioBuffer((int)size, slotBuffer)).apply(width * height * 3);

        int slot = 0;
        for (BufferedImage srcItem : srcList) {
            slot++;
            double[] pixels = srcItem.getRaster().getPixels(0, 0, width, height, (double[]) null);

            for (int k = 0; k < pixels.length; k++) {
                Indice key = pixelMapper.getIndice(k, slot);
                dm.put(key, dm.mapper(pixels[k]));
            }

        }

        if (Objects.nonNull(responceEscale)) {
            dm = responceEscale.apply(dm);
        }
        return dm;
    }

    @Override
    public NumericMatriz<N> getOutputLayer() {
        return outputLayer;
    }

    @Override
    public void startProduction() {
        outputLayer = scanInput();
        for (LayerConsumer<N> lc : consumers) {
            lc.setInputLayer(outputLayer);
            lc.layerComplete(LayerConsumer.SUCCESS_STATUS);
        }
    }

    @Override
    public List<LayerConsumer<N>> getConsumers() {
        return consumers;
    }

    public void setSrc(BufferedImage src) {
        this.src = src;
    }

    public List<BufferedImage> getSrcList() {
        return srcList;
    }

    public int getSlotBuffer() {
        return slotBuffer;
    }

}
