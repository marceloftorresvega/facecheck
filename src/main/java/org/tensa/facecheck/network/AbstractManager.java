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
 * @param <N>
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
    
    protected Boolean[] useBias;

    /**
     * Get the value of useBias
     *
     * @return the value of useBias
     */
    public Boolean[] isUseBias() {
        return useBias;
    }

    /**
     * Set the value of useBias
     *
     * @param useBias new value of useBias
     */
    public void setUseBias(Boolean[] useBias) {
        this.useBias = useBias;
    }

    /**
     * Get the value of useBias at specified index
     *
     * @param index the index of useBias
     * @return the value of useBias at specified index
     */
    public Boolean isUseBias(int index) {
        return this.useBias[index];
    }

    /**
     * Set the value of useBias at specified index.
     *
     * @param index the index of useBias
     * @param useBias new value of useBias at specified index
     */
    public void setUseBias(int index, Boolean useBias) {
        this.useBias[index] = useBias;
    }


    public AbstractManager() {
    }

    public Activation<N>[] getActivationFunction() {
        return activationFunction;
    }

    /**
     * funciones de activacion de cada capa
     * @param activationFunction 
     */
    public void setActivationFunction(Activation<N>[] activationFunction) {
        this.activationFunction = activationFunction;
    }

    public Activation<N> getActivationFunction(int index) {
        return this.activationFunction[index];
    }

    /**
     * funcion de activacion de una capa determinada
     * @param index
     * @param activationFunction 
     */
    public void setActivationFunction(int index, Activation<N> activationFunction) {
        this.activationFunction[index] = activationFunction;
    }

    public N[] getLearningRate() {
        return learningRate;
    }

    /**
     * tasa de aprendisaje
     * @param learningRate 
     */
    public void setLearningRate(N[] learningRate) {
        this.learningRate = learningRate;
    }

    public N getLearningRate(int index) {
        return this.learningRate[index];
    }

    /**
     * tasa de aprendisaje de una determinada capa
     * @param index
     * @param learningRate 
     */
    public void setLearningRate(int index, N learningRate) {
        this.learningRate[index] = learningRate;
    }

    public LearningEstrategy<N>[] getLearningControl() {
        return learningControl;
    }

    /**
     * estrategia de aprendisaje de las capas
     * @param learningControl 
     */
    public void setLearningControl(LearningEstrategy<N>[] learningControl) {
        this.learningControl = learningControl;
    }

    public LearningEstrategy<N> getHiddenLearningGuide(int index) {
        return this.learningControl[index];
    }

    /**
     * estrategia de aprendisaje de determinada capa
     * @param index
     * @param hiddenLearningGuide 
     */
    public void setHiddenLearningGuide(int index, LearningEstrategy<N> hiddenLearningGuide) {
        this.learningControl[index] = hiddenLearningGuide;
    }

    public int[] getHiddenStep() {
        return hiddenStep;
    }

    /**
     * ubicacion de la progracion para learning rate
     * @param hiddenStep 
     */
    public void setHiddenStep(int[] hiddenStep) {
        this.hiddenStep = hiddenStep;
    }

    public int getHiddenStep(int index) {
        return this.hiddenStep[index];
    }

    /**
     * ubicacion de la progracion para learning rate de la capa
     * @param index
     * @param hiddenStep 
     */
    public void setHiddenStep(int index, int hiddenStep) {
        this.hiddenStep[index] = hiddenStep;
    }

    public NumericMatriz<N>[] getWeights() {
        return weights;
    }

    /**
     * pesos de conexion de todas las capas
     * @param weights 
     */
    public void setWeights(NumericMatriz<N>[] weights) {
        this.weights = weights;
    }

    public NumericMatriz<N> getWeights(int index) {
        return this.weights[index];
    }

    /**
     * pesos de conexion de determinada capa
     * @param index
     * @param weights 
     */
    public void setWeights(int index, NumericMatriz<N> weights) {
        this.weights[index] = weights;
    }

    public NumericMatriz<N> getErrorGraph() {
        return errorGraph;
    }

    /**
     * grafico de errores
     * @param errorGraph 
     */
    public void setErrorGraph(NumericMatriz<N> errorGraph) {
        this.errorGraph = errorGraph;
    }

    /**
     * carga matrices de pesos de las capas desde un archivo
     * @param archivo nombre del archivo
     */
    @SuppressWarnings("unchecked")
    public void cargaPesos(String archivo) {
        log.info("cargaPesos <{}>", archivo);
        try (final InputStream fis = Files.newInputStream(Paths.get(archivo)); final BufferedInputStream bis = new BufferedInputStream(fis); final GzipCompressorInputStream gzIn = new GzipCompressorInputStream(bis); final ObjectInputStream ois = new ObjectInputStream(gzIn)) {
            weights = (NumericMatriz<N>[]) ois.readObject();

            inStep = weights[0].getDominio().getColumna();
            log.info("neuronas <{}>", inStep);
            hiddenStep = Arrays.stream(weights).map(NumericMatriz::getDominio).mapToInt(Dominio::getFila).peek(hid -> log.info("neuronas <{}>", hid)).toArray();
        } catch (FileNotFoundException ex) {
            log.error("error al cargar pesos", ex);
        } catch (IOException | ClassNotFoundException ex) {
            log.error("error al cargar pesos", ex);
        }
    }

    /**
     * salva matrices de pesos de las capas en un archivo
     * @param archivo nombre del archivo
     */
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

    /**
     * valor de secuencia para learning rate
     * @param inStep 
     */
    public void setInStep(int inStep) {
        this.inStep = inStep;
    }

    /**
     * crea matriz de pesos de las conexiones de las neuronas de una capa completa
     * @param innerSize the value of innerSize
     * @param outerSize the value of outerSize
     * @param creating the value of creating
     * @param modeling the value of modeling
     * @return NumericMatriz
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
     * iniciacion de matricess de pesos basados en los parametros
     * @param creation inicial de neuronas de capas
     * @param modeling funcion de modelado de neurona
     */
    @SuppressWarnings("unchecked")
    public void initMatrix(UnaryOperator<NumericMatriz<N>>[] creation, UnaryOperator<NumericMatriz<N>>[] modeling) {
        int inSize = pixelMapper.getDominio(inStep).getFila();
        weights = new NumericMatriz[hiddenStep.length];
        for (int k = 0; k < weights.length; k++) {
            inSize = this.useBias[k]? inSize +1: inSize;
            weights[k] = createMatrix(inSize, hiddenStep[k], creation[k], modeling[k]);
            inSize = hiddenStep[k];
        }
    }

    /**
     * relaciona capas ocultas con transferencia de delta por back propagation
     * @param origen capa oculta de origen
     * @param destino capa oculta de consicutiva
     */
    protected void relate(HiddenLayer<N> origen, HiddenLayer<N> destino) {
        origen.getConsumers().add(destino);
        destino.getProducers().add(origen);
    }

    /**
     * relaciona capa productora con consumidora sin traspaso de delta
     * @param origen capa productora
     * @param destino capa consumidora
     */
    protected void relate(LayerProducer<N> origen, LayerConsumer<N> destino) {
        origen.getConsumers().add(destino);
    }

    /**
     * relaciona capa con capa de diferencia para calculo de delta para back propagation
     * @param origen capa de origen
     * @param terminal capa comparadora
     */
    protected void relate(HiddenLayer<N> origen, DiffLayer<N> terminal) {
        origen.getConsumers().add(terminal.getInternalBridgeConsumer());
        terminal.getProducers().add(origen);
    }

    /**
     * funcion de presentacion de error
     * @param learning valor delta transferido
     * @param idx epoc
     */
    protected void errorBiConsumer(LayerLearning<N> learning, ParOrdenado idx) {
        N errorVal = learning.getError().get(Indice.D1);
        synchronized (errorGraph) {
            errorGraph.put(idx, errorGraph.mapper(errorVal.doubleValue()));
        }
        log.info("diferencia <{}>", errorVal);
    }

    /**
     * generador de matrices
     * @param supplier procedimiento de generacion de matrices
     */
    public void setSupplier(Function<Dominio, NumericMatriz<N>> supplier) {
        this.supplier = supplier;
    }

    /**
     * operacion de adaptacion de entrada de la red
     * @param inputScale operador
     */
    public void setInputScale(UnaryOperator<NumericMatriz<N>> inputScale) {
        this.inputScale = inputScale;
    }

    /**
     * imagen de entrada
     * @param inputImage
     */
    public void setInputImage(BufferedImage inputImage) {
        this.inputImage = inputImage;
    }

    /**
     * imagen deseada a producir
     * @param compareImage
     */
    public void setCompareImage(BufferedImage compareImage) {
        this.compareImage = compareImage;
    }

    /**
     * imagen de salida
     * @return BufferedImage
     */
    public BufferedImage getOutputImage() {
        return outputImage;
    }

    public void setOutputImage(BufferedImage outputImage) {
        this.outputImage = outputImage;
    }

    /**
     * modalidad de entrenamiento true|false
     * @return boolean
     */
    public boolean isTrainingMode() {
        return trainingMode;
    }

    /**
     * modalidad de entrenamiento
     * @param trainingMode true|false
     */
    public void setTrainingMode(boolean trainingMode) {
        this.trainingMode = trainingMode;
    }

    /**
     * cantidad de iteraciones
     * @return numero
     */
    public int getIterateTo() {
        return iterateTo;
    }

    /**
     * cantidad de iteraciones
     * @param iterateTo numero de iteraciones
     */
    public void setIterateTo(int iterateTo) {
        this.iterateTo = iterateTo;
    }

    /**
     * estado activacion de detencion de emergencia
     * @return true|false
     */
    public boolean isEmergencyBreak() {
        return emergencyBreak;
    }

    /**
     * activacion de detencion de emergencia
     * @param emergencyBreak true|false
     */
    public void setEmergencyBreak(boolean emergencyBreak) {
        this.emergencyBreak = emergencyBreak;
    }

    /**
     * se activó uso de seleccion para entrenamiento
     * @return true|false
     */
    public boolean isUseSelection() {
        return useSelection;
    }

    /**
     * activar uso de seleccion para entrenamiento
     * @param useSelection true|false
     */
    public void setUseSelection(boolean useSelection) {
        this.useSelection = useSelection;
    }

    /**
     * cola de areas de entrenamiento
     * @return lista 
     */
    public LinkedList<Rectangle> getAreaQeue() {
        return areaQeue;
    }

    /**
     * indice de entrada
     * @return ParOrdnado
     */
    public List<ParOrdenado> getProccesDomain() {
        return proccesDomain;
    }

    /**
     * iteracion actual
     * @return numero
     */
    public int getIterateCurrent() {
        return iterateCurrent;
    }

    /**
     * modelo de lectura de pixels
     * @param pixelMapper 
     */
    public void setPixelMapper(PixelMapper pixelMapper) {
        this.pixelMapper = pixelMapper;
    }

    /**
     * procesamiento de la red neuronal
     */
    public abstract void process();
}
