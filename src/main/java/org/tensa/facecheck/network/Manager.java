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
package org.tensa.facecheck.network;

import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.IOException;
import java.util.function.BooleanSupplier;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.tensa.facecheck.activation.impl.LinealActivationImpl;
import org.tensa.facecheck.activation.impl.SigmoidActivationImpl;
import org.tensa.facecheck.layer.impl.BackDoorLayer;
import org.tensa.facecheck.layer.impl.DiffLayer;
import org.tensa.facecheck.layer.impl.DoorLayer;
import org.tensa.facecheck.layer.impl.HiddenLayer;
import org.tensa.facecheck.layer.impl.NormalizeLayer;
import org.tensa.facecheck.layer.impl.OutputScale;
import org.tensa.facecheck.layer.impl.PixelInputLayer;
import org.tensa.facecheck.layer.impl.PixelOutputLayer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 * back propagation inicial
 *
 * @author Marcelo
 */
public class Manager<N extends Number> extends AbstractManager<N> {

    public Manager(Function<Dominio, NumericMatriz<N>> supplier, int inStep, int outStep, int hidStep, BufferedImage outputImage, BufferedImage inputImage, BufferedImage compareImage, int iterateTo) {
        this.supplier = supplier;
        this.inStep = inStep;
        this.outStep = outStep;
        this.hidStep = hidStep;
        this.outputImage = outputImage;
        this.inputImage = inputImage;
        this.compareImage = compareImage;
        this.iterateTo = iterateTo;
    }

    public Manager(Function<Dominio, NumericMatriz<N>> supplier) {
        this.supplier = supplier;
    }

    public Manager() {
    }

    @Override
    public void process() {

        log.info("iniciando proceso...");

        int width = inputImage.getWidth();
        int height = inputImage.getHeight();

        log.info("procesando... {} {}", width - inStep, height - inStep);

        for (iterateCurrent = 0; (!emergencyBreak) && ((!trainingMode) && iterateCurrent < 1 || trainingMode && iterateCurrent < ((Integer) iterateTo)); iterateCurrent++) {

            log.info("iteracion <{}>", iterateCurrent);
            Dominio dominio = new Dominio(width - inStep, height - inStep);

            hiddenLearningRate = hiddenLearningControl.updateFactor(iterateCurrent, hiddenLearningRate);
            outputLearningRate = outputLearningControl.updateFactor(iterateCurrent, outputLearningRate);

            proccesDomain = dominio.stream()
                    .filter(idx -> (((idx.getFila() - (inStep - outStep) / 2) % outStep == 0) && ((idx.getColumna() - (inStep - outStep) / 2) % outStep == 0)))
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
                        HiddenLayer<N> hiddenLayer = new HiddenLayer<>(weightsH, hiddenLearningRate, new SigmoidActivationImpl<>());
                        NormalizeLayer<N> normaLayer = new NormalizeLayer<>();
                        HiddenLayer<N> learnLayer = new HiddenLayer<>(weightsO, outputLearningRate, new LinealActivationImpl<>());
                        PixelInputLayer<N> simplePixelsCompareLayer = new PixelInputLayer<>(supplier, pixelMapper, OutputScale::scale);
                        PixelOutputLayer<N> pixelsOutputLayer = new PixelOutputLayer<>(pixelMapper);
                        DiffLayer<N> diffLAyer = new DiffLayer<>(simplePixelsCompareLayer, (lL) -> {
                            errorBiConsumer(lL, idx);
                        });
                        DiffLayer<N> middleTest = new DiffLayer<>(normaLayer, (lL) -> {
                            errorBiConsumer(lL, idx);
                        });

                        BooleanSupplier compareLayerExpresion = () -> {
                            return trainingMode && (middleTest.getPropagationError() == null || (middleTest.getError().get(Indice.D1).doubleValue() > .05));
                        };
                        BackDoorLayer<N> backIfLayer = new BackDoorLayer<>(compareLayerExpresion);

                        DoorLayer<N> ifLayer = new DoorLayer<>(compareLayerExpresion);

                        relate(simplePixelsInputLayer, hiddenLayer);

                        relate(hiddenLayer, middleTest);
//                            hiddenLayer.getConsumers().remove(middleTest);
                        middleTest.getProducers().remove(hiddenLayer);

                        if (trainingMode) {
                            backIfLayer.getProducers().add(hiddenLayer);
                            middleTest.getProducers().add(backIfLayer);
                        }

                        relate(hiddenLayer, ifLayer);
//                            relate(ifLayer,middleTest.getInternalBridgeConsumer());
                        relate(ifLayer.getElseProducer(), learnLayer);

                        relate(hiddenLayer, learnLayer);
                        hiddenLayer.getConsumers().remove(learnLayer);

//                            relate(hiddenLayer, learnLayer);
//                            hiddenLayer.getConsumers().remove(learnLayer);
//                            relate(hiddenLayer, normaLayer);
//                            relate(normaLayer, learnLayer);
                        relate(learnLayer, pixelsOutputLayer);

                        try {
                            //                    log.info("cargando bloque ejecucion <{}><{}>", i, j);
                            BufferedImage dest = outputImage.getSubimage(i + (inStep - outStep) / 2, j + (inStep - outStep) / 2, outStep, outStep);
                            pixelsOutputLayer.setDest(dest);

                            BufferedImage src = inputImage.getSubimage(i, j, inStep, inStep);
                            simplePixelsInputLayer.setSrc(src);

                            //                        log.info("cargando bloque comparacion <{}><{}>", i, j);
                            BufferedImage comp = compareImage.getSubimage(i + (inStep - outStep) / 2, j + (inStep - outStep) / 2, outStep, outStep);
                            simplePixelsCompareLayer.setSrc(comp);
                        } catch(java.awt.image.RasterFormatException ex ) {
                            emergencyBreak = true;
                            ex.printStackTrace();
                        }

                        if (trainingMode) {
                            relate(learnLayer, diffLAyer);

                        }

                        simplePixelsInputLayer.startProduction();

                        try (
                                Closeable inpmat = simplePixelsInputLayer.getOutputLayer();
                                Closeable outmat = simplePixelsCompareLayer.getOutputLayer();
                                Closeable hiddProp = hiddenLayer.getPropagationError();
                                Closeable hiddOut = hiddenLayer.getOutputLayer();
                                Closeable learnProp = learnLayer.getPropagationError();
                                Closeable learnOut = learnLayer.getOutputLayer();) {

                        } catch (IOException ex) {
                            //clear
                        }
                    });

        }

    }

}
