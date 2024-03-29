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
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.mapping.PixelMapper;
import org.tensa.facecheck.mapping.PixelMappings;
import org.tensa.tensada.matrix.NumericMatriz;

/**
 *
 * @author lorenzo
 * @param <N>
 */
public class PixelOutputLayer<N extends Number> implements LayerConsumer<N> {
    protected int status;
    protected NumericMatriz<N> inputLayer;
    protected BufferedImage dest;
    private List<BufferedImage> destList;
    protected final PixelMapper pixelMapper;

    public PixelOutputLayer() {
        pixelMapper = PixelMappings.defaultMapping();
        destList = new ArrayList<>();
    }

    public PixelOutputLayer(PixelMapper pixelMapper) {
        this.pixelMapper = pixelMapper;
        destList = new ArrayList<>();
    }

    @Override
    public NumericMatriz<N> setInputLayer(NumericMatriz<N> inputLayer) {
        NumericMatriz<N> last = this.inputLayer;
        this.inputLayer = inputLayer;
        return last;
    }

    @Override
    public void layerComplete(int status) {
        this.status = status;
        if (status == LayerConsumer.SUCCESS_STATUS) {
            
            int slot =0;
            for (BufferedImage myDest : destList) {
                slot++;

                double[] pixels = new double[pixelMapper.getLargo(inputLayer.getDominio())];
                for( int i =0; i< pixels.length; i++) {
                    pixels[i] = 255 * inputLayer.get(pixelMapper.getIndice(i, slot)).doubleValue();
                }
                int width = myDest.getWidth();
                int height = myDest.getHeight();
                myDest.getRaster().setPixels(0, 0, width, height, pixels);
            }
            
        }
    }

    public BufferedImage getDest() {
        return dest;
    }

    public void setDest(BufferedImage dest) {
        this.dest = dest;
        this.destList.add(dest);
    }

    public List<BufferedImage> getDestList() {
        return destList;
    }

    public void setDestList(List<BufferedImage> destList) {
        this.destList = destList;
    }
    
}
