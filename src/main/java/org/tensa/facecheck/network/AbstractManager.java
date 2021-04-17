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
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.activation.Activation;
import org.tensa.facecheck.layer.LayerConsumer;
import org.tensa.facecheck.layer.LayerLearning;
import org.tensa.facecheck.layer.LayerProducer;
import org.tensa.facecheck.layer.impl.DiffLayer;
import org.tensa.facecheck.layer.impl.HiddenLayer;
import org.tensa.facecheck.mapping.PixelMapper;
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
    protected NumericMatriz<N> errorGraph;
    protected Function<Dominio, NumericMatriz<N>> supplier;
    protected UnaryOperator<NumericMatriz<N>> inputScale;
    protected PixelMapper pixelMapper;
    protected int inStep;
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
    protected NumericMatriz<N>[] weights;
    protected int[] hiddenStep;
    protected LearningEstrategy<N>[] learningControl;
    protected N[] learningRate;
    protected Activation<N>[] activationFunction;

    public AbstractManager() {
    }

    public Activation<N>[] getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(Activation<N>[] activationFunction) {
        this.activationFunction = activationFunction;
    }

    public Activation<N> getActivationFunction(int index) {
        return this.activationFunction[index];
    }

    public void setActivationFunction(int index, Activation<N> activationFunction) {
        this.activationFunction[index] = activationFunction;
    }

    public N[] getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(N[] learningRate) {
        this.learningRate = learningRate;
    }

    public N getLearningRate(int index) {
        return this.learningRate[index];
    }

    public void setLearningRate(int index, N learningRate) {
        this.learningRate[index] = learningRate;
    }

    public LearningEstrategy<N>[] getLearningControl() {
        return learningControl;
    }

    public void setLearningControl(LearningEstrategy<N>[] learningControl) {
        this.learningControl = learningControl;
    }

    public LearningEstrategy<N> getHiddenLearningGuide(int index) {
        return this.learningControl[index];
    }

    public void setHiddenLearningGuide(int index, LearningEstrategy<N> hiddenLearningGuide) {
        this.learningControl[index] = hiddenLearningGuide;
    }

    public int[] getHiddenStep() {
        return hiddenStep;
    }

    public void setHiddenStep(int[] hiddenStep) {
        this.hiddenStep = hiddenStep;
    }

    public int getHiddenStep(int index) {
        return this.hiddenStep[index];
    }

    public void setHiddenStep(int index, int hiddenStep) {
        this.hiddenStep[index] = hiddenStep;
    }

    public NumericMatriz<N>[] getWeights() {
        return weights;
    }

    public void setWeights(NumericMatriz<N>[] weights) {
        this.weights = weights;
    }

    public NumericMatriz<N> getWeights(int index) {
        return this.weights[index];
    }

    public void setWeights(int index, NumericMatriz<N> weights) {
        this.weights[index] = weights;
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
            weights = (NumericMatriz<N>[]) ois.readObject();

            inStep = weights[0].getDominio().getColumna();
            log.info("neuronas <{}>", inStep);
            hiddenStep = Arrays.stream(weights).map(NumericMatriz::getDominio).mapToInt(Dominio::getFila).peek(hid -> log.info("neuronas <{}>", hid)).toArray();
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
            oos.writeObject(weights);
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

    /**
     *
     * @param innerSize the value of innerSize
     * @param outerSize the value of outerSize
     * @param creating the value of creating
     * @param modeling the value of modeling
     */
    public NumericMatriz<N> createMatrix(int innerSize, int outerSize, UnaryOperator<NumericMatriz<N>> creating, UnaryOperator<NumericMatriz<N>> modeling) {
        log.info("iniciando 1..<{},{}>", outerSize, innerSize);
        try (final Matriz<Dominio> dominioMatriz = new Matriz<>(new Dominio(outerSize, 1))) {
            dominioMatriz.getDominio().forEach(idx -> {
                dominioMatriz.put(idx, new Dominio(1, innerSize));
            });
            log.info("iniciando 2..<{},{}>", outerSize, innerSize);
            return creating.compose(supplier).andThen((NumericMatriz<N> m) -> {
                BlockMatriz<N> hiddenBlockMatriz = BlockMatriz.wrapper(dominioMatriz, m);
                hiddenBlockMatriz.forEach((ix, sm) -> {
                    NumericMatriz<N> nm = supplier.apply(sm.getDominio());
                    nm.putAll(sm);
                    NumericMatriz<N> fm = modeling.apply(nm);
                    sm.putAll(fm);
                    nm.clear();
                    fm.clear();
                });
                return m;
            }).apply(new Dominio(outerSize, innerSize));
        } catch (IOException ex) {
            log.error("createMatrix", ex);
            throw new RuntimeException(ex);
        }
    }

    /**
     *
     * @param modeling the value of modelingH
     * @param modelingO the value of modelingO
     */
    public void initMatrix(UnaryOperator<NumericMatriz<N>>[] creation, UnaryOperator<NumericMatriz<N>>[] modeling) {
        int inSize = pixelMapper.getDominio(inStep).getFila();
        weights = new NumericMatriz[hiddenStep.length];
        for (int k = 0; k < weights.length; k++) {
            weights[k] = createMatrix(inSize, hiddenStep[k], creation[k], modeling[k]);
            inSize = hiddenStep[k];
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

    public void setPixelMapper(PixelMapper pixelMapper) {
        this.pixelMapper = pixelMapper;
    }

    public abstract void process();
}
