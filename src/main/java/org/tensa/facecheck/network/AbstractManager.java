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
package org.tensa.facecheck.network;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
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
import java.util.function.Function;
import java.util.function.UnaryOperator;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.facecheck.layer.impl.DiffLayer;
import org.tensa.facecheck.layer.impl.HiddenLayer;
import org.tensa.facecheck.mapping.PixelMapper;
import org.tensa.facecheck.weight.WeightCreationStyle;
import org.tensa.tensada.matrix.BlockMatriz;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.Matriz;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 * abstract manager para backpropagation
 *
 * @author Marcelo
 */
public abstract class AbstractManager<N extends Number> {

    protected final Logger log = LoggerFactory.getLogger(AbstractManager.class);
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

    public AbstractManager() {
    }

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
        try (final InputStream fis = Files.newInputStream(Paths.get(archivo)); final BufferedInputStream bis = new BufferedInputStream(fis); final GzipCompressorInputStream gzIn = new GzipCompressorInputStream(bis); final ObjectInputStream ois = new ObjectInputStream(gzIn)) {
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
        try (final OutputStream fos = Files.newOutputStream(Paths.get(archivo)); final BufferedOutputStream out = new BufferedOutputStream(fos); final GzipCompressorOutputStream gzOut = new GzipCompressorOutputStream(out); final ObjectOutputStream oos = new ObjectOutputStream(gzOut)) {
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
                hiddenBlockMatriz.put(idx, supplier.apply(new Dominio(1, innerSize)));
            });
            log.info("iniciando 2..<{},{}>", outerSize, innerSize);
            return creating.compose(supplier).andThen((NumericMatriz<N> m) -> {
                hiddenBlockMatriz.splitIn(m);
                hiddenBlockMatriz.replaceAll((ParOrdenado idx, Matriz<N> sm) -> modeling.apply((NumericMatriz<N>) sm));
                return hiddenBlockMatriz.build();
            }).andThen((Matriz<N> m) -> {
                return supplier.andThen((NumericMatriz<N> fm) -> {
                    fm.putAll(m);
                    return fm;
                }).apply(m.getDominio());
            }).apply(new Dominio(outerSize, innerSize));
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
        log.info("diferencia <{}>", errorVal);
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

    public abstract void process();
}
