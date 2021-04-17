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
package org.tensa.facecheck.network.impl;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.facecheck.layer.impl.DiffLayer;
import org.tensa.facecheck.layer.impl.HiddenLayer;
import org.tensa.facecheck.layer.impl.OutputScale;
import org.tensa.facecheck.layer.impl.PixelInputLayer;
import org.tensa.facecheck.layer.impl.PixelOutputLayer;
import org.tensa.facecheck.network.AbstractManager;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 * back propagation inicial
 *
 * @author Marcelo
 * @param <N>
 */
public class ManagerBackPropImpl<N extends Number> extends AbstractManager<N> {

    public ManagerBackPropImpl(Function<Dominio, NumericMatriz<N>> supplier, int inStep, BufferedImage outputImage, BufferedImage inputImage, BufferedImage compareImage, int iterateTo) {
        this.supplier = supplier;
        this.inStep = inStep;
        this.outputImage = outputImage;
        this.inputImage = inputImage;
        this.compareImage = compareImage;
        this.iterateTo = iterateTo;
    }

    public ManagerBackPropImpl(Function<Dominio, NumericMatriz<N>> supplier) {
        this.supplier = supplier;
    }

    public ManagerBackPropImpl() {
    }

    @Override
    public void process() {

        log.info("iniciando proceso...");

        final int width = inputImage.getWidth();
        final int height = inputImage.getHeight();

        final int rInStep = (int) Math.sqrt(inStep / 3);
        final int rOutStep = (int) Math.sqrt(hiddenStep[hiddenStep.length - 1] / 3);
        final int rDeltaStep = rInStep - rOutStep;

        log.info("procesando... {} {}", width - rInStep, height - rInStep);

        for (iterateCurrent = 0; (!emergencyBreak) && ((!trainingMode) && iterateCurrent < 1 || trainingMode && iterateCurrent < ((Integer) iterateTo)); iterateCurrent++) {

            log.info("iteracion <{}>", iterateCurrent);
            Dominio dominio = new Dominio(width - (int) rInStep, height - (int) rInStep);

            for (int k = 0; k < weights.length; k++) {
                learningRate[k] = learningControl[k].updateFactor(iterateCurrent, learningRate[k]);
            }

            proccesDomain = dominio.stream()
                    .filter(idx -> (((idx.getFila() - rDeltaStep / 2) % rOutStep == 0) && ((idx.getColumna() - rDeltaStep / 2) % rOutStep == 0)))
                    .filter(idx -> (!useSelection) || (areaQeue.stream().anyMatch(a -> a.contains(idx.getFila(), idx.getColumna()))))
                    .collect(Collectors.toList());
            errorGraph = supplier.apply(dominio);
            proccesDomain.stream()
                    .sorted((idx1, idx2) -> (int) (2.0 * Math.random() - 1.0))
                    .parallel()
                    .filter(idx -> !emergencyBreak)
                    .forEach((ParOrdenado idx) -> {
                        int i = idx.getFila();
                        int j = idx.getColumna();

                        PixelInputLayer<N> simplePixelsInputLayer = new PixelInputLayer<>(supplier, pixelMapper, inputScale);
                        HiddenLayer<N>[] hiddenLayers = new HiddenLayer[weights.length];
                        for (int k = 0; k < weights.length; k++) {
                            hiddenLayers[k] = new HiddenLayer<>(weights[k], learningRate[k], activationFunction[k]);
                        }

                        PixelInputLayer<N> simplePixelsCompareLayer = new PixelInputLayer<>(supplier, pixelMapper, OutputScale::scale);
                        PixelOutputLayer<N> pixelsOutputLayer = new PixelOutputLayer<>(pixelMapper);
                        DiffLayer<N> diffLAyer = new DiffLayer<>(simplePixelsCompareLayer, (lL) -> {
                            errorBiConsumer(lL, idx);
                        });

                        relate(simplePixelsInputLayer, hiddenLayers[0]);
                        for (int k = 1; k < weights.length; k++) {
                            relate(hiddenLayers[k - 1], hiddenLayers[k]);

                        }
                        relate(hiddenLayers[weights.length - 1], pixelsOutputLayer);

                        try {
                            //                    log.info("cargando bloque ejecucion <{}><{}>", i, j);
                            BufferedImage dest = outputImage.getSubimage(i + rDeltaStep / 2, j + rDeltaStep / 2, rOutStep, rOutStep);
                            pixelsOutputLayer.setDest(dest);

                            BufferedImage src = inputImage.getSubimage(i, j, rInStep, rInStep);
                            simplePixelsInputLayer.setSrc(src);

                            //                        log.info("cargando bloque comparacion <{}><{}>", i, j);
                            BufferedImage comp = compareImage.getSubimage(i + rDeltaStep / 2, j + rDeltaStep / 2, rOutStep, rOutStep);
                            simplePixelsCompareLayer.setSrc(comp);
                        } catch (java.awt.image.RasterFormatException ex) {
                            emergencyBreak = true;
                            ex.printStackTrace();
                        }

                        if (trainingMode) {
                            relate(hiddenLayers[weights.length - 1], diffLAyer);

                        }

                        simplePixelsInputLayer.startProduction();

                        Stream.of(simplePixelsInputLayer, simplePixelsCompareLayer)
                                .map(LayerProducer::getOutputLayer)
                                .filter(m -> Objects.nonNull(m))
                                .forEach(NumericMatriz::clear);

                        Arrays.stream(hiddenLayers)
                                .flatMap(l -> Stream.of(l.getPropagationError(), l.getOutputLayer()))
                                .filter(m -> Objects.nonNull(m))
                                .forEach(NumericMatriz::clear);
                    });

        }

    }

}
