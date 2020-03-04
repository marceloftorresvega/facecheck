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
import java.util.stream.Collectors;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.activation.impl.HiddenSigmoidActivationImpl;
import org.tensa.facecheck.activation.impl.LinealActivationImpl;
import org.tensa.facecheck.layer.impl.HiddenLayer;
import org.tensa.facecheck.layer.impl.OutputScale;
import org.tensa.facecheck.layer.impl.PixelInputLayer;
import org.tensa.facecheck.layer.impl.PixelOutputLayer;
import org.tensa.tensada.matrix.BlockMatriz;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.Matriz;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 * back propagation inicial
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
    
    private NumericMatriz<N> weightsH;
    private NumericMatriz<N> weightsO;
    private NumericMatriz<N> errorGraph;
    private Function<Dominio,NumericMatriz<N>> supplier;
    private UnaryOperator<NumericMatriz<N>> inputScale;
    
    private int inStep;
    private int outStep;
    private int hidStep;
    
    private N hiddenLearningRate;
    private N outputLearningRate;
    
    private BufferedImage outputImage;
    private BufferedImage inputImage ;
    private BufferedImage compareImage ;
    
    private final LinkedList<Rectangle> areaQeue = new LinkedList<>();
    private List<ParOrdenado> proccesDomain;
    
    private boolean trainingMode;
    private int iterateTo;
    private boolean emergencyBreak;
    private int iterateCurrent;
    
    private boolean useSelection;
    
        
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
        log.info("cargaPesos <{}>",archivo);
        try (
                InputStream fis = Files.newInputStream(Paths.get(archivo));
                BufferedInputStream bis = new BufferedInputStream(fis);
                GzipCompressorInputStream gzIn = new GzipCompressorInputStream(bis);
                ObjectInputStream ois = new ObjectInputStream(gzIn)
                ) {
            weightsH = (NumericMatriz<N>)ois.readObject();
            Integer fila = weightsH.getDominio().getFila();
            Integer columna = weightsH.getDominio().getColumna();
            
            weightsO = (NumericMatriz<N>)ois.readObject();
            fila = weightsO.getDominio().getFila();
            
            inStep = (int) Math.sqrt(columna/3);
            hidStep = fila;
            outStep = (int) Math.sqrt(fila/3);
            
        } catch ( FileNotFoundException ex) {
            log.error("error al cargar pesos", ex);
        } catch (IOException ex) {
            log.error("error al cargar pesos", ex);
        } catch (ClassNotFoundException ex) {
            log.error("error al cargar pesos", ex);
        }
    }
    
    public void salvaPesos(String archivo) {
        log.info("salvaPesos <{}>", archivo);

         try( 
                 OutputStream fos = Files.newOutputStream(Paths.get(archivo));
                 BufferedOutputStream out = new BufferedOutputStream(fos);
                 GzipCompressorOutputStream gzOut = new GzipCompressorOutputStream(out);
                 ObjectOutputStream oos = new ObjectOutputStream(gzOut); ) {
            oos.writeObject(weightsH);
            oos.writeObject(weightsO);
             
         }catch (FileNotFoundException ex) {
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
    
    public NumericMatriz<N> createMatrix(int innerSize, int outerSize, UnaryOperator<NumericMatriz<N>> creation) {
        
        log.info("iniciando 1..<{},{}>",outerSize, innerSize);
        try (BlockMatriz<N> hiddenBlockMatriz = new BlockMatriz<>(new Dominio(outerSize, 1))) {
            hiddenBlockMatriz.getDominio().forEach((ParOrdenado idx) -> {
                final NumericMatriz<N> tmpm = supplier.apply(new Dominio(1, innerSize));
                tmpm.getDominio().forEach((i) -> {
                    tmpm.put(i, tmpm.mapper(1-2*Math.random()));
                });
                hiddenBlockMatriz.put(idx, creation.apply(tmpm));
                tmpm.clear();
            });

            log.info("iniciando 2..<{},{}>",outerSize, innerSize);
            Matriz<N> merged = hiddenBlockMatriz.merge();

            return supplier.apply(new Dominio(Indice.D1))
                    .instancia(merged.getDominio(), merged);
            
        } catch (IOException ex) {
            
            log.error("createMatrix", ex);
            throw new RuntimeException(ex);
        }
    }
    
    public void initMatrix( UnaryOperator<NumericMatriz<N>> creationH, UnaryOperator<NumericMatriz<N>> creationO){
        
        int inSize = inStep*inStep*3;
        int outSize = outStep*outStep*3;
        
        weightsH = createMatrix(inSize, hidStep, creationH);
        weightsO = createMatrix(hidStep, outSize, creationO);
        
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
            
            log.info("procesando...");

            for(iterateCurrent=0; (!emergencyBreak) && ((!trainingMode) && iterateCurrent<1 || trainingMode && iterateCurrent<((Integer) iterateTo)); iterateCurrent++) {

                log.info("iteracion <{}>", iterateCurrent);
                Dominio dominio = new Dominio(width-inStep, height-inStep);
                
                proccesDomain = dominio.stream()
                        .filter( idx -> (( (idx.getFila()-(inStep-outStep)/2) % outStep ==0) && ((idx.getColumna()-(inStep-outStep)/2)% outStep == 0)))
                        .filter(idx -> (!useSelection) || ( areaQeue.stream().anyMatch(a -> a.contains(idx.getFila(), idx.getColumna()))) )
                        .collect(Collectors.toList());
                errorGraph = supplier.apply(dominio);
                proccesDomain.stream()
                        .sorted((idx1,idx2) -> (int)(2.0*Math.random()-1.0))
                        .parallel()
                        .filter(idx -> !emergencyBreak)
                        .forEach((ParOrdenado idx) -> {
                            int i = idx.getFila();
                            int j = idx.getColumna();

                            PixelInputLayer<N> simplePixelsInputLayer = new PixelInputLayer<>(null, supplier, inputScale);
                            HiddenLayer<N> hiddenLayer = new HiddenLayer<>(weightsH,  hiddenLearningRate,new HiddenSigmoidActivationImpl<>());
                            HiddenLayer<N> pixelLeanringLayer = new HiddenLayer<>(weightsO, outputLearningRate, new LinealActivationImpl<>());
                            PixelInputLayer<N> simplePixelsCompareLayer = new PixelInputLayer<>(null, supplier, OutputScale::scale);
                            PixelOutputLayer<N> pixelsOutputLayer = new PixelOutputLayer<>();

                            relate(simplePixelsInputLayer, hiddenLayer);
                            relate(hiddenLayer, pixelLeanringLayer);
                            relate(pixelLeanringLayer, pixelsOutputLayer);

        //                    log.info("cargando bloque ejecucion <{}><{}>", i, j);
                            pixelsOutputLayer.setDest(outputImage.getSubimage(i + (inStep-outStep)/2, j + (inStep-outStep)/2, outStep, outStep));
                            BufferedImage src = inputImage.getSubimage(i, j, inStep, inStep);
                            simplePixelsInputLayer.setSrc(src);
                            simplePixelsInputLayer.startProduction();

                            if(trainingMode){
        //                        log.info("cargando bloque comparacion <{}><{}>", i, j);
                                BufferedImage comp = compareImage.getSubimage(i + (inStep-outStep)/2, j + (inStep-outStep)/2, outStep, outStep);
                                
                                simplePixelsCompareLayer.setSrc(comp);
                                simplePixelsCompareLayer.startProduction();
                                pixelLeanringLayer.setLearningData(simplePixelsCompareLayer.getOutputLayer());

                                pixelLeanringLayer.startLearning();
                                N errorVal = pixelLeanringLayer.getError().get(Indice.D1);
                                    
                                synchronized(errorGraph) {
                                    errorGraph.put(idx, errorGraph.mapper(errorVal.doubleValue()));
                                }
                                log.info("diferencia <{}>", errorVal);                                
                                
                                simplePixelsCompareLayer.getOutputLayer().clear();
                                hiddenLayer.getPropagationError().clear();
                                pixelLeanringLayer.getPropagationError().clear();
                            }
                            simplePixelsInputLayer.getOutputLayer().clear();
                            hiddenLayer.getOutputLayer().clear();
                            pixelLeanringLayer.getOutputLayer().clear();
                        });

            }
        
    }
    
    private void relate(HiddenLayer<N> origen, HiddenLayer<N> destino) {
        origen.getConsumers().add(destino);
        destino.getProducers().add(origen);
        
    }
    
    private void relate(PixelInputLayer<N> origen, HiddenLayer<N> destino) {
        origen.getConsumers().add(destino);
        
    }
    
    private void relate(HiddenLayer<N> origen, PixelOutputLayer<N> destino) {
        origen.getConsumers().add(destino);
        
    }

    public void setSupplier(Function<Dominio,NumericMatriz<N>> supplier) {
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
}
