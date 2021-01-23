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

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.activation.impl.LinealActivationImpl;
import org.tensa.facecheck.activation.impl.SigmoidActivationImpl;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.facecheck.layer.impl.BackDoorLayer;
import org.tensa.facecheck.layer.impl.DiffLayer;
import org.tensa.facecheck.layer.impl.DoorLayer;
import org.tensa.facecheck.layer.impl.HiddenLayer;
import org.tensa.facecheck.layer.impl.NormalizeLayer;
import org.tensa.facecheck.layer.impl.OutputScale;
import org.tensa.facecheck.layer.impl.PixelInputLayer;
import org.tensa.facecheck.layer.impl.PixelOutputLayer;
import org.tensa.facecheck.weight.WeightCreationStyle;
import org.tensa.facecheck.mapping.PixelMapper;
import org.tensa.tensada.matrix.BlockMatriz;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 * back propagation inicial
 *
 * @author Marcelo
 */
public class Manager<N extends Number> {

    private final Logger log = LoggerFactory.getLogger(Manager.class);

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

    protected NumericMatriz<N> weightsH;
    protected NumericMatriz<N> weightsO;
    protected NumericMatriz<N> errorGraph;
    protected Function<Dominio, NumericMatriz<N>> supplier;
    protected UnaryOperator<NumericMatriz<N>> inputScale;
    protected PixelMapper pixelMapper;

    protected int inStep;
    protected int outStep;
    protected int hidStep;

    protected LearningControl<N> hiddenLearningControl;
    protected LearningControl<N> outputLearningControl;
    protected N hiddenLearningRate;
    protected N outputLearningRate;

    protected BufferedImage outputImage;
    protected BufferedImage inputImage;
    protected BufferedImage compareImage;

    protected final LinkedList<Rectangle> areaQeue = new LinkedList<>();
    protected List<ParOrdenado> proccesDomain;

    protected boolean trainingMode;
    protected int iterateTo;
    protected boolean emergencyBreak;
    protected int iterateCurrent;

    protected boolean useSelection;

    public NumericMatriz<N> getWeightsH() {
        return weightsH;
    }

    public void setWeightsH(NumericMatriz<N> weightsH) {
        this.weightsH = weightsH;
    }

    public NumericMatriz<N> getWeightsO() {
        return weightsO;
    }

    public void setWeightsO(NumericMatriz<N> weightsO) {
        this.weightsO = weightsO;
    }

    public NumericMatriz<N> getErrorGraph() {
        return errorGraph;
    }

    public void setErrorGraph(NumericMatriz<N> errorGraph) {
        this.errorGraph = errorGraph;
    }

    public void cargaPesos(String archivo) {
        log.info("cargaPesos <{}>", archivo);
        try (
                InputStream fis = Files.newInputStream(Paths.get(archivo));
                BufferedInputStream bis = new BufferedInputStream(fis);
                GzipCompressorInputStream gzIn = new GzipCompressorInputStream(bis);
                ObjectInputStream ois = new ObjectInputStream(gzIn)) {
            weightsH = (NumericMatriz<N>) ois.readObject();
            Integer fila = weightsH.getDominio().getFila();
            Integer columna = weightsH.getDominio().getColumna();

            weightsO = (NumericMatriz<N>) ois.readObject();
            Integer filao = weightsO.getDominio().getFila();

            inStep = (int) Math.sqrt(columna / 3);
            hidStep = fila;
            outStep = (int) Math.sqrt(filao / 3);

        } catch (FileNotFoundException ex) {
            log.error("error al cargar pesos", ex);
        } catch (IOException ex) {
            log.error("error al cargar pesos", ex);
        } catch (ClassNotFoundException ex) {
            log.error("error al cargar pesos", ex);
        }
    }

    public void salvaPesos(String archivo) {
        log.info("salvaPesos <{}>", archivo);

        try (
                OutputStream fos = Files.newOutputStream(Paths.get(archivo));
                BufferedOutputStream out = new BufferedOutputStream(fos);
                GzipCompressorOutputStream gzOut = new GzipCompressorOutputStream(out);
                ObjectOutputStream oos = new ObjectOutputStream(gzOut);) {
            oos.writeObject(weightsH);
            oos.writeObject(weightsO);

        } catch (FileNotFoundException ex) {
            log.error("error al guardar  pesos", ex);
        } catch (IOException ex) {
            log.error("error al guardar  pesos", ex);
        }
    }

    public int getInStep() {
        return inStep;
    }

    public void setInStep(int inStep) {
        this.inStep = inStep;
    }

    public int getOutStep() {
        return outStep;
    }

    public void setOutStep(int outStep) {
        this.outStep = outStep;
    }

    public int getHidStep() {
        return hidStep;
    }

    public void setHidStep(int hidStep) {
        this.hidStep = hidStep;
    }

    /**
     *
     * @param innerSize the value of innerSize
     * @param outerSize the value of outerSize
     * @param creating the value of creating
     * @param modeling the value of modeling
     */
    public NumericMatriz<N> createMatrix(int innerSize, int outerSize, UnaryOperator<NumericMatriz<N>> creating, UnaryOperator<NumericMatriz<N>> modeling) {
        log.info("iniciando 1..<{},{}>", outerSize, innerSize);
        try (final BlockMatriz<N> hiddenBlockMatriz = new BlockMatriz<>(new Dominio(outerSize, 1))) {
            hiddenBlockMatriz.getDominio().forEach((ParOrdenado idx) -> {
                hiddenBlockMatriz.put(idx, supplier
                        .apply(new Dominio(1, innerSize)));
            });

            log.info("iniciando 2..<{},{}>", outerSize, innerSize);
            return creating.compose(supplier)
                    .andThen((m) -> {
                        hiddenBlockMatriz.splitIn(m);
                        hiddenBlockMatriz.replaceAll((idx, sm) -> modeling.apply((NumericMatriz<N>) sm));
                        return hiddenBlockMatriz.build();
                    }).andThen((m) -> {
                return supplier.andThen((fm) -> {
                    fm.putAll(m);
                    return fm;
                })
                        .apply(m.getDominio());
            })
                    .apply(new Dominio(outerSize, innerSize));

        } catch (IOException ex) {

            log.error("createMatrix", ex);
            throw new RuntimeException(ex);
        }
    }

    /**
     *
     * @param modelingH the value of modelingH
     * @param modelingO the value of modelingO
     */
    public void initMatrix(UnaryOperator<NumericMatriz<N>> modelingH, UnaryOperator<NumericMatriz<N>> modelingO) {

        int inSize = pixelMapper.getDominio(inStep * inStep * 3).getFila();
        int outSize = pixelMapper.getDominio(outStep * outStep * 3).getFila();

        weightsH = createMatrix(inSize, hidStep, WeightCreationStyle::randomCreationStyle, modelingH);
        weightsO = createMatrix(hidStep, outSize, WeightCreationStyle::randomCreationStyle, modelingO);

    }

    public N getHiddenLearningRate() {
        return hiddenLearningRate;
    }

    public void setHiddenLearningRate(N hiddenLearningRate) {
        this.hiddenLearningRate = hiddenLearningRate;
    }

    public N getOutputLearningRate() {
        return outputLearningRate;
    }

    public void setOutputLearningRate(N outputLearningRate) {
        this.outputLearningRate = outputLearningRate;
    }

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

                        //                    log.info("cargando bloque ejecucion <{}><{}>", i, j);
                        BufferedImage dest = outputImage.getSubimage(i + (inStep - outStep) / 2, j + (inStep - outStep) / 2, outStep, outStep);
                        pixelsOutputLayer.setDest(dest);

                        BufferedImage src = inputImage.getSubimage(i, j, inStep, inStep);
                        simplePixelsInputLayer.setSrc(src);

                        //                        log.info("cargando bloque comparacion <{}><{}>", i, j);
                        BufferedImage comp = compareImage.getSubimage(i + (inStep - outStep) / 2, j + (inStep - outStep) / 2, outStep, outStep);
                        simplePixelsCompareLayer.setSrc(comp);

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

    protected void relate(HiddenLayer<N> origen, HiddenLayer<N> destino) {
        origen.getConsumers().add(destino);
        destino.getProducers().add(origen);

    }

    protected void relate(LayerProducer<N> origen, LayerConsumer<N> destino) {
        origen.getConsumers().add(destino);

    }

    /**
     *
     * @param origen the value of regreso
     * @param terminal the value of origen
     */
    protected void relate(HiddenLayer<N> origen, DiffLayer<N> terminal) {
        origen.getConsumers().add(terminal.getInternalBridgeConsumer());
        terminal.getProducers().add(origen);

    }

    protected void errorBiConsumer(LayerLearning<N> learning, ParOrdenado idx) {
        N errorVal = learning.getError().get(Indice.D1);

        synchronized (errorGraph) {
            errorGraph.put(idx, errorGraph.mapper(errorVal.doubleValue()));
        }
//        log.info("diferencia <{}>", errorVal); 

    }

    public void setSupplier(Function<Dominio, NumericMatriz<N>> supplier) {
        this.supplier = supplier;
    }

    public void setInputScale(UnaryOperator<NumericMatriz<N>> inputScale) {
        this.inputScale = inputScale;
    }

    public void setInputImage(BufferedImage inputImage) {
        this.inputImage = inputImage;
    }

    public void setCompareImage(BufferedImage compareImage) {
        this.compareImage = compareImage;
    }

    public BufferedImage getOutputImage() {
        return outputImage;
    }

    public void setOutputImage(BufferedImage outputImage) {
        this.outputImage = outputImage;
    }

    public boolean isTrainingMode() {
        return trainingMode;
    }

    public void setTrainingMode(boolean trainingMode) {
        this.trainingMode = trainingMode;
    }

    public int getIterateTo() {
        return iterateTo;
    }

    public void setIterateTo(int iterateTo) {
        this.iterateTo = iterateTo;
    }

    public boolean isEmergencyBreak() {
        return emergencyBreak;
    }

    public void setEmergencyBreak(boolean emergencyBreak) {
        this.emergencyBreak = emergencyBreak;
    }

    public boolean isUseSelection() {
        return useSelection;
    }

    public void setUseSelection(boolean useSelection) {
        this.useSelection = useSelection;
    }

    public LinkedList<Rectangle> getAreaQeue() {
        return areaQeue;
    }

    public List<ParOrdenado> getProccesDomain() {
        return proccesDomain;
    }

    public int getIterateCurrent() {
        return iterateCurrent;
    }

    public void setHiddenLearningControl(LearningControl<N> hiddenLearningControl) {
        this.hiddenLearningControl = hiddenLearningControl;
    }

    public void setOutputLearningControl(LearningControl<N> outputLearningControl) {
        this.outputLearningControl = outputLearningControl;
    }

    public void setPixelMapper(PixelMapper pixelMapper) {
        this.pixelMapper = pixelMapper;
    }
}
