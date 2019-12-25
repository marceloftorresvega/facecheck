package org.tensa.facecheck.main;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ConvolveOp;
import java.awt.image.IndexColorModel;
import java.awt.image.Kernel;
import java.awt.image.WritableRaster;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;
import javax.swing.ComboBoxModel;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JPanel;
import javax.swing.SpinnerListModel;
import javax.swing.SpinnerModel;
import javax.swing.SpinnerNumberModel;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.filter.MaskOp;
import org.tensa.facecheck.layer.impl.HiddenSigmoidLayer;
import org.tensa.facecheck.layer.impl.PixelDirectSigmoidLeanringLayer;
import org.tensa.facecheck.layer.impl.PixelsDirectSigmoidOutputLayer;
import org.tensa.facecheck.layer.impl.PixelsDirectInputLayer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.DoubleMatriz;
import org.tensa.tensada.matrix.Indice;
import org.tensa.tensada.matrix.ParOrdenado;

/**
 *
 * @author lorenzo
 */
public class VisLoad extends javax.swing.JFrame {

    private final Logger log = LoggerFactory.getLogger(VisLoad.class);

    private final String baseUrl = "\\img\\originales\\";
    private final String testBaseUrl = "\\img\\procesadas\\";
    private final String weightUrl = "\\data\\";
    
    private final String[] imageName = {"IMG_2869", "IMG_2918","IMG_3071","IMG_3076","IMG_3078","IMG_3079"};
    
    private final Double[] learningFactor = {.001, 0.003, .004, .005, .008, .01, .03, .04, .05, .08, .1, .3, .4, .5, .8};

    private final String sufxType = ".jpg";
    private ComboBoxModel comboModel;
    private BufferedImage buffImage ;
    private BufferedImage destBuffImage ;
    private final int kwidth = 27;
    private float[] data;
    private BufferedImage bufferImageFiltered;
    private DoubleMatriz weightsH;
    private DoubleMatriz weightsO;
    private int inStep;
    private int outStep;
    private int hidStep;
    private Rectangle learnArea;
    private LinkedList<Rectangle> areaQeue;
    private boolean areaDelete = false;
    private boolean areaSelect = false;

    /**
     * Get the value of comboModel
     *
     * @return the value of comboModel
     */
    public ComboBoxModel getComboModel() {
        if(Objects.isNull(comboModel))
            comboModel = new DefaultComboBoxModel(imageName);
            
        return comboModel;
      
    }
    
    public SpinnerModel getSpinnerModel(){
//        if(Objects.isNull(spinnerModel))
        SpinnerListModel spinnerModel = new SpinnerListModel(learningFactor);
        return spinnerModel;
    }

    /**
     * Get the value of sufxType
     *
     * @return the value of sufxType
     */
    public String getSufxType() {
        return sufxType;
    }

    /**
     * Get the value of imageName
     *
     * @return the value of imageName
     */
    public String[] getImageName() {
        return imageName;
    }

    /**
     * Get the value of imageName at specified index
     *
     * @param index the index of imageName
     * @return the value of imageName at specified index
     */
    public String getImageName(int index) {
        return this.imageName[index];
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    /**
     * Creates new form visLoad
     */
    public VisLoad() {
        initComponents();
        data = new float[kwidth * kwidth];
        float total =0;
        
        for(int i = 0; i < kwidth * kwidth; i++){
            data[i] = calculaMatriz(i % kwidth, i / kwidth);
            total+= data[i];
            
        }
        
        for(int i =0; i < kwidth * kwidth;i++){
            data[i] /= total / 1.5; 
        }
        learnArea = new Rectangle();
        areaQeue = new LinkedList<>();
        areaQeue.add(learnArea);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jSplitPane1 = new javax.swing.JSplitPane();
        vista = getNuevaVista();
        respuesta = getNuevaRespuesta();
        jPanel6 = new javax.swing.JPanel();
        jTabbedPane1 = new javax.swing.JTabbedPane();
        jPanel1 = new javax.swing.JPanel();
        cargaImagen = new javax.swing.JButton();
        jComboBox1 = new javax.swing.JComboBox();
        jPanel2 = new javax.swing.JPanel();
        suavizaResultado = new javax.swing.JButton();
        enmascaraResultado = new javax.swing.JButton();
        cargaOriginal = new javax.swing.JButton();
        cargaPreparada = new javax.swing.JButton();
        jSeparator1 = new javax.swing.JSeparator();
        jPanel3 = new javax.swing.JPanel();
        cargar = new javax.swing.JButton();
        clean = new javax.swing.JButton();
        salva = new javax.swing.JButton();
        jLabel1 = new javax.swing.JLabel();
        inNeurs = new javax.swing.JSpinner();
        jLabel2 = new javax.swing.JLabel();
        hiddNeurs = new javax.swing.JSpinner();
        jLabel3 = new javax.swing.JLabel();
        outNeurs = new javax.swing.JSpinner();
        jPanel4 = new javax.swing.JPanel();
        procesar = new javax.swing.JButton();
        entrenar = new javax.swing.JCheckBox();
        hiddenLearningRate = new javax.swing.JSpinner();
        outputLearningRate = new javax.swing.JSpinner();
        iteraciones = new javax.swing.JSpinner();
        jCheckBox1 = new javax.swing.JCheckBox();
        jButton3 = new javax.swing.JButton();
        freno = new javax.swing.JToggleButton();
        actualizacion = new javax.swing.JCheckBox();
        jPanel5 = new javax.swing.JPanel();
        seleccion = new javax.swing.JCheckBox();
        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jButton4 = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Vista de carga");

        jSplitPane1.setDividerLocation(128);

        vista.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        vista.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                vistaMouseClicked(evt);
            }
            public void mouseReleased(java.awt.event.MouseEvent evt) {
                vistaMouseReleased(evt);
            }
        });

        javax.swing.GroupLayout vistaLayout = new javax.swing.GroupLayout(vista);
        vista.setLayout(vistaLayout);
        vistaLayout.setHorizontalGroup(
            vistaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );
        vistaLayout.setVerticalGroup(
            vistaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 116, Short.MAX_VALUE)
        );

        jSplitPane1.setLeftComponent(vista);

        respuesta.setBorder(javax.swing.BorderFactory.createEmptyBorder(1, 1, 1, 1));

        javax.swing.GroupLayout respuestaLayout = new javax.swing.GroupLayout(respuesta);
        respuesta.setLayout(respuestaLayout);
        respuestaLayout.setHorizontalGroup(
            respuestaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );
        respuestaLayout.setVerticalGroup(
            respuestaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 116, Short.MAX_VALUE)
        );

        jSplitPane1.setRightComponent(respuesta);

        cargaImagen.setText("Carga...");
        cargaImagen.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cargaImagenActionPerformed(evt);
            }
        });

        jComboBox1.setModel(getComboModel());

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(cargaImagen)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jComboBox1, javax.swing.GroupLayout.PREFERRED_SIZE, 131, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(422, 422, 422))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cargaImagen)
                    .addComponent(jComboBox1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        jTabbedPane1.addTab("Imagen", null, jPanel1, "Carga de imagen");

        suavizaResultado.setText("Suaviza");
        suavizaResultado.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                suavizaResultadoActionPerformed(evt);
            }
        });

        enmascaraResultado.setText("enmascara");
        enmascaraResultado.setToolTipText("mascara de salida preparada sobre original");
        enmascaraResultado.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                enmascaraResultadoActionPerformed(evt);
            }
        });

        cargaOriginal.setText("Original");
        cargaOriginal.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cargaOriginalActionPerformed(evt);
            }
        });

        cargaPreparada.setText("Preparada");
        cargaPreparada.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cargaPreparadaActionPerformed(evt);
            }
        });

        jSeparator1.setOrientation(javax.swing.SwingConstants.VERTICAL);

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(cargaOriginal)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(cargaPreparada)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jSeparator1, javax.swing.GroupLayout.PREFERRED_SIZE, 15, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(suavizaResultado)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(enmascaraResultado)
                .addContainerGap(354, Short.MAX_VALUE))
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(suavizaResultado)
                    .addComponent(enmascaraResultado)
                    .addComponent(cargaOriginal)
                    .addComponent(cargaPreparada))
                .addGap(0, 0, Short.MAX_VALUE))
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jSeparator1)
                .addContainerGap())
        );

        jTabbedPane1.addTab("Pre proceso salida", null, jPanel2, "Pre proceso de imagen de salida");

        cargar.setText("Carga");
        cargar.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cargarActionPerformed(evt);
            }
        });

        clean.setText("Limpiar");
        clean.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cleanActionPerformed(evt);
            }
        });

        salva.setText("Salva");
        salva.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                salvaActionPerformed(evt);
            }
        });

        jLabel1.setText("Entrada");

        inNeurs.setModel(new javax.swing.SpinnerNumberModel(101, 3, 1000, 1));
        inNeurs.setToolTipText("Neuronas de entrada (pixels)");

        jLabel2.setText("Oculta");

        hiddNeurs.setModel(new javax.swing.SpinnerNumberModel(15, 3, 100, 1));
        hiddNeurs.setToolTipText("Neuronas ocultas");

        jLabel3.setText("Salida");

        outNeurs.setModel(new javax.swing.SpinnerNumberModel(101, 3, 1000, 1));
        outNeurs.setToolTipText("Neuronas de salida (pixels)");

        javax.swing.GroupLayout jPanel3Layout = new javax.swing.GroupLayout(jPanel3);
        jPanel3.setLayout(jPanel3Layout);
        jPanel3Layout.setHorizontalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(clean)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(cargar)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(salva)
                .addGap(18, 18, 18)
                .addComponent(jLabel1)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(inNeurs, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jLabel2)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(hiddNeurs, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jLabel3)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(outNeurs, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(214, Short.MAX_VALUE))
        );
        jPanel3Layout.setVerticalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(clean)
                    .addComponent(cargar)
                    .addComponent(salva)
                    .addComponent(jLabel1)
                    .addComponent(inNeurs, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel2)
                    .addComponent(hiddNeurs, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel3)
                    .addComponent(outNeurs, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(0, 0, Short.MAX_VALUE))
        );

        jTabbedPane1.addTab("Pesos", null, jPanel3, "administracion de pesos");

        procesar.setText("Procesar");
        procesar.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                procesarActionPerformed(evt);
            }
        });

        entrenar.setSelected(true);
        entrenar.setText("entrenar");
        entrenar.setToolTipText("modo de proceso");

        hiddenLearningRate.setModel(getSpinnerModel());
        hiddenLearningRate.setToolTipText("de capa oculta");

        outputLearningRate.setModel(getSpinnerModel());
        outputLearningRate.setToolTipText("de capa de salida");

        iteraciones.setModel(new javax.swing.SpinnerNumberModel(50, 1, 500, 10));
        iteraciones.setToolTipText("Iteraciones");

        jCheckBox1.setText("Usa Selección");
        jCheckBox1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jCheckBox1ActionPerformed(evt);
            }
        });

        jButton3.setText("Limpiar");
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });

        freno.setText("Freno");

        actualizacion.setText("Continua");
        actualizacion.setToolTipText("actualizacion de pantalla cada 30 segundos");

        javax.swing.GroupLayout jPanel4Layout = new javax.swing.GroupLayout(jPanel4);
        jPanel4.setLayout(jPanel4Layout);
        jPanel4Layout.setHorizontalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel4Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(procesar)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(freno)
                .addGap(12, 12, 12)
                .addComponent(entrenar)
                .addGap(18, 18, 18)
                .addComponent(hiddenLearningRate, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(outputLearningRate, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(iteraciones, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(jCheckBox1)
                .addGap(18, 18, 18)
                .addComponent(jButton3)
                .addGap(18, 18, 18)
                .addComponent(actualizacion)
                .addContainerGap(100, Short.MAX_VALUE))
        );
        jPanel4Layout.setVerticalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel4Layout.createSequentialGroup()
                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(entrenar)
                    .addComponent(procesar)
                    .addComponent(hiddenLearningRate, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(outputLearningRate, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(iteraciones, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jCheckBox1)
                    .addComponent(jButton3)
                    .addComponent(freno)
                    .addComponent(actualizacion))
                .addGap(0, 0, Short.MAX_VALUE))
        );

        jTabbedPane1.addTab("Entrenamiento", jPanel4);

        seleccion.setText("Usa selección");
        seleccion.setActionCommand("Usar Seleccion");
        seleccion.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                seleccionActionPerformed(evt);
            }
        });

        jButton1.setText("Agrega selección");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jButton2.setText("Quita selección");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        jButton4.setText("Modifica selección");
        jButton4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton4ActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel5Layout = new javax.swing.GroupLayout(jPanel5);
        jPanel5.setLayout(jPanel5Layout);
        jPanel5Layout.setHorizontalGroup(
            jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel5Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(seleccion)
                .addGap(18, 18, 18)
                .addComponent(jButton1)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jButton2)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jButton4)
                .addContainerGap(312, Short.MAX_VALUE))
        );
        jPanel5Layout.setVerticalGroup(
            jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel5Layout.createSequentialGroup()
                .addGroup(jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(seleccion)
                    .addComponent(jButton1)
                    .addComponent(jButton2)
                    .addComponent(jButton4))
                .addGap(0, 0, Short.MAX_VALUE))
        );

        jTabbedPane1.addTab("Seleccion", jPanel5);

        javax.swing.GroupLayout jPanel6Layout = new javax.swing.GroupLayout(jPanel6);
        jPanel6.setLayout(jPanel6Layout);
        jPanel6Layout.setHorizontalGroup(
            jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel6Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jTabbedPane1)
                .addContainerGap())
        );
        jPanel6Layout.setVerticalGroup(
            jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel6Layout.createSequentialGroup()
                .addGap(0, 0, 0)
                .addComponent(jTabbedPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 57, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap())
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jSplitPane1)
                    .addComponent(jPanel6, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel6, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jSplitPane1)
                .addContainerGap())
        );

        getAccessibleContext().setAccessibleName("Oculta Caras");

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void cargaImagenActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cargaImagenActionPerformed
        String _filename1 = System.getProperty("user.dir") + baseUrl + (String)jComboBox1.getSelectedItem() + sufxType;
        String _filename2 = System.getProperty("user.dir") + testBaseUrl + (String)jComboBox1.getSelectedItem() + sufxType;
        log.info("directorio user <{}>",System.getProperty("user.dir"));
        
        try {
            buffImage = ImageIO.read(new File(_filename1));
            
            destBuffImage = ImageIO.read(new File(_filename2));
            bufferImageFiltered = destBuffImage;
            java.awt.EventQueue.invokeLater(() -> {
                vista.repaint();
                respuesta.repaint();
            });
        } catch (IOException ex) {
            log.error("error de archivo <{}>", _filename1, ex);
        }
    }//GEN-LAST:event_cargaImagenActionPerformed

    private void suavizaResultadoActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_suavizaResultadoActionPerformed
        
        
        ConvolveOp conv = new ConvolveOp(new Kernel(kwidth, kwidth, data));
        if(Objects.nonNull(bufferImageFiltered))
            bufferImageFiltered.flush();
        
        bufferImageFiltered = conv.filter(destBuffImage, null);
        destBuffImage = bufferImageFiltered;
        
        java.awt.EventQueue.invokeLater(() -> {
            respuesta.repaint();
        });
    }//GEN-LAST:event_suavizaResultadoActionPerformed

    private void enmascaraResultadoActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_enmascaraResultadoActionPerformed
        
        MaskOp conv = new MaskOp();
        conv.setOtherSrc(destBuffImage);
        if(Objects.nonNull(bufferImageFiltered))
            bufferImageFiltered.flush();
        
        bufferImageFiltered = conv.filter(buffImage, null);
        destBuffImage = bufferImageFiltered;
        
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
            respuesta.repaint();
        });
    }//GEN-LAST:event_enmascaraResultadoActionPerformed

    private void procesarActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_procesarActionPerformed

        log.info("iniciando 3...");
        bufferImageFiltered = createCompatibleDestImage(buffImage, null);
        
        int width = buffImage.getWidth();
        int height = buffImage.getHeight();
        
        log.info("procesando...");
        new Thread(() -> {
            procesar.setEnabled(false);
            jButton3.setEnabled(false);
            clean.setEnabled(false);
        
            for(int idIteracion=0; (!freno.isSelected()) && ((!entrenar.isSelected()) && idIteracion<1 || entrenar.isSelected() && idIteracion<((Integer) iteraciones.getValue())); idIteracion++) {

                log.info("iteracion <{}>", idIteracion);
                new Dominio(width-inStep, height-inStep).stream()
                        .filter( idx -> (( (idx.getFila()-(inStep-outStep)/2) % outStep ==0) && ((idx.getColumna()-(inStep-outStep)/2)% outStep == 0)))
                        .filter(idx -> (!seleccion.isSelected()) || ( areaQeue.stream().anyMatch(a -> a.contains(idx.getFila(), idx.getColumna()))) )
                        .sorted((idx1,idx2) -> (int)(2.0*Math.random()-1.0))
                        .parallel()
                        .filter(idx -> !freno.isSelected())
                        .forEach(idx -> {
                            int i = idx.getFila();
                            int j = idx.getColumna();

                            PixelsDirectInputLayer simplePixelsInputLayer = new PixelsDirectInputLayer();
                            PixelsDirectInputLayer simplePixelsCompareLayer = new PixelsDirectInputLayer();
                            HiddenSigmoidLayer hiddenLayer = new HiddenSigmoidLayer(weightsH,  (Double)hiddenLearningRate.getValue());
                            PixelDirectSigmoidLeanringLayer pixelLeanringLayer = new PixelDirectSigmoidLeanringLayer(weightsO, (Double)outputLearningRate.getValue());
                            PixelsDirectSigmoidOutputLayer pixelsOutputLayer = new PixelsDirectSigmoidOutputLayer(null);

                            simplePixelsInputLayer.getConsumers().add(hiddenLayer);
                            hiddenLayer.getConsumers().add(pixelLeanringLayer);
                            pixelLeanringLayer.getConsumers().add(pixelsOutputLayer);

        //                    log.info("cargando bloque ejecucion <{}><{}>", i, j);
                            pixelsOutputLayer.setDest(bufferImageFiltered.getSubimage(i + (inStep-outStep)/2, j + (inStep-outStep)/2, outStep, outStep));
                            BufferedImage src = buffImage.getSubimage(i, j, inStep, inStep);
                            simplePixelsInputLayer.setSrc(src);
                            simplePixelsInputLayer.startProduction();

                            if(entrenar.isSelected()){
        //                        log.info("cargando bloque comparacion <{}><{}>", i, j);
                                BufferedImage comp = destBuffImage.getSubimage(i + (inStep-outStep)/2, j + (inStep-outStep)/2, outStep, outStep);
                                simplePixelsCompareLayer.setSrc(comp);
                                simplePixelsCompareLayer.startProduction();
                                pixelLeanringLayer.setCompareToLayer(simplePixelsCompareLayer.getOutputLayer());

                                pixelLeanringLayer.adjustBack();
                                log.info("diferencia <{}>", pixelLeanringLayer.getError().get(Indice.D1));
                            }
                        });

            }
            
            java.awt.EventQueue.invokeLater(() -> {
                respuesta.repaint();
            });
            
            procesar.setEnabled(true);
            jButton3.setEnabled(true);
            clean.setEnabled(true);
            freno.setSelected(false);
        }).start();
        
        new Thread( () -> {
            while (!procesar.isEnabled()) {
                try {
                    Thread.sleep(30000);
                    if (actualizacion.isSelected()) {
                        synchronized(respuesta){
//                        java.awt.EventQueue.invokeLater(() -> {
                            respuesta.repaint();
                            log.info("realiza actualizacion");
//                        });
                        }
                        
                    } else {
                        log.info("no realiza actualizacion");
                    }
                } catch (InterruptedException ex) {
                   log.error("error en actualizador", ex);
                }
            }
            log.info("finaliza actualizador");
            
        }).start();
    }//GEN-LAST:event_procesarActionPerformed

    private void cleanActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cleanActionPerformed
        log.info("iniciando 0...");
        inStep = (Integer)inNeurs.getValue();
        hidStep = (Integer)hiddNeurs.getValue();
        outStep = (Integer)outNeurs.getValue();
        
        int inSize = inStep*inStep*3;
        int outSize = outStep*outStep*3;
        
        log.info("iniciando 1.0..<{},{}>",hidStep, inSize);
        weightsH = (DoubleMatriz)new DoubleMatriz(new Dominio(hidStep, inSize)).matrizUno();
        log.info("iniciando 1.1..<{},{}>",hidStep, inSize);
        weightsH.replaceAll((ParOrdenado i, Double v) -> 0.5-Math.random() );
//        log.info("iniciando 1.2..<{},{}>",hidStep, inSize);
//        weightsH = (DoubleMatriz)weightsH.productoEscalar( hidStep / Math.sqrt(weightsH.distanciaE2().get(Indice.D1)) );
        
        log.info("iniciando 2.0..<{},{}>",outSize, hidStep);
        weightsO = (DoubleMatriz)new DoubleMatriz(new Dominio(outSize, hidStep)).matrizUno();
        log.info("iniciando 2.1..<{},{}>",outSize, hidStep);
        weightsO.replaceAll((ParOrdenado i, Double v) -> 0.5-Math.random() );
//        log.info("iniciando 2.2..<{},{}>",outSize, hidStep);
//        weightsO = (DoubleMatriz)weightsO.productoEscalar( hidStep/ Math.sqrt(weightsO.distanciaE2().get(Indice.D1)) );
        
    }//GEN-LAST:event_cleanActionPerformed

    private void salvaActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_salvaActionPerformed
        String filename = System.getProperty("user.dir") + weightUrl + "nw.dat";
        
         try( 
                 OutputStream fos = Files.newOutputStream(Paths.get(filename));
                 BufferedOutputStream out = new BufferedOutputStream(fos);
                 GzipCompressorOutputStream gzOut = new GzipCompressorOutputStream(out);
                 DataOutputStream dos = new DataOutputStream(gzOut)
                 )   {
            
            Integer fila = weightsH.getDominio().getFila();
            Integer columna = weightsH.getDominio().getColumna();
            
            dos.writeInt(fila);
            dos.writeInt(columna);
            
            List<ParOrdenado> listado = weightsH.getDominio()
                    .stream()
                    .sorted(this::compareTo)
                    .collect(Collectors.toList());
            
            for ( ParOrdenado indice : listado) {
                dos.writeInt(indice.getFila());
                dos.writeInt(indice.getColumna());
                dos.writeDouble(weightsH.get(indice));
            }
            
            fila = weightsO.getDominio().getFila();
            columna = weightsO.getDominio().getColumna();
            
            dos.writeInt(fila);
            dos.writeInt(columna);
            
            listado = weightsO.getDominio()
                    .stream()
                    .sorted(this::compareTo)
                    .collect(Collectors.toList());
            
            for ( ParOrdenado indice : listado) {
                dos.writeInt(indice.getFila());
                dos.writeInt(indice.getColumna());
                dos.writeDouble(weightsO.get(indice));
                
            }
            
         } catch (FileNotFoundException ex) {
             log.error("error al guardar  pesos", ex);
         } catch (IOException ex) {
             log.error("error al guardar  pesos", ex);             
         }
    }//GEN-LAST:event_salvaActionPerformed

    private void cargarActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cargarActionPerformed
        String filename = System.getProperty("user.dir") + weightUrl + "nw.dat";
        
        try (
                InputStream fis = Files.newInputStream(Paths.get(filename));
                BufferedInputStream bis = new BufferedInputStream(fis);
                GzipCompressorInputStream gzIn = new GzipCompressorInputStream(bis);
                DataInputStream dis = new DataInputStream(gzIn)
                ) {

            Integer fila;
            Integer columna;

            fila = dis.readInt();
            columna = dis.readInt();
            
            inNeurs.setValue((int)Math.sqrt(columna/3));
            hiddNeurs.setValue(fila);
            
            log.info("leer <{}>, <{}>", fila, columna);
            Dominio dominio = new Dominio(fila, columna);
            
            weightsH = new DoubleMatriz(dominio);
            
            List<ParOrdenado> listado = weightsH.getDominio()
                    .stream()
                    .sorted(this::compareTo)
                    .collect(Collectors.toList());
            for ( ParOrdenado indice : listado) {
                weightsH.indexa(dis.readInt(), dis.readInt(), dis.readDouble());
                
            }
            
            fila = dis.readInt();
            columna = dis.readInt();
            
            outNeurs.setValue((int)Math.sqrt(fila/3));
                        
            log.info("leer <{}>, <{}>", fila, columna);
            dominio = new Dominio(fila, columna);
            
            weightsO = new DoubleMatriz(dominio);
            
            listado = weightsO.getDominio()
                    .stream()
                    .sorted(this::compareTo)
                    .collect(Collectors.toList());
            
            for ( ParOrdenado indice : listado) {
                
                weightsO.indexa(dis.readInt(), dis.readInt(), dis.readDouble());
                
            }
            
            inStep = (Integer)inNeurs.getValue();
            hidStep = (Integer)hiddNeurs.getValue();
            outStep = (Integer)outNeurs.getValue();
            
        } catch ( FileNotFoundException ex) {
            log.error("error al cargar pesos", ex);
        } catch (IOException ex) {
            log.error("error al cargar pesos", ex);
        }
    }//GEN-LAST:event_cargarActionPerformed

    private void vistaMouseReleased(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_vistaMouseReleased
        if (!areaDelete && !areaSelect) {
            int x = evt.getX();
            int y = evt.getY();
            float escala = (float)buffImage.getWidth() / (float)vista.getBounds().width;

            learnArea.width = (int) (x * escala) - learnArea.x;
            learnArea.height = (int) (y * escala) - learnArea.y;

            java.awt.EventQueue.invokeLater(() -> {
                vista.repaint();
            });
        }
    }//GEN-LAST:event_vistaMouseReleased

    private void vistaMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_vistaMouseClicked
        int x = evt.getX();
        int y = evt.getY();
        
        float escala = (float)buffImage.getWidth() / (float)vista.getBounds().width;
        if (areaDelete && !areaSelect) {
            areaQeue.stream()
                    .filter( a -> a.contains((int) (x * escala), (int) (y * escala)))
                    .findFirst()
                    .ifPresent( a -> areaQeue.removeFirstOccurrence(a));
            areaDelete = false;
            learnArea = areaQeue.getLast();
        } else if ( areaSelect && !areaDelete) {
            areaQeue.stream()
                    .filter( a -> a.contains((int) (x * escala), (int) (y * escala)))
                    .findFirst()
                    .ifPresent( a -> learnArea = a);
            areaSelect = false;
        } else {
            learnArea.x = (int) (x * escala);
            learnArea.y = (int) (y * escala);
        }
        
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
        });
    }//GEN-LAST:event_vistaMouseClicked

    private void jCheckBox1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jCheckBox1ActionPerformed
        seleccion.setSelected(jCheckBox1.isSelected());
    }//GEN-LAST:event_jCheckBox1ActionPerformed

    private void seleccionActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_seleccionActionPerformed
        jCheckBox1.setSelected(seleccion.isSelected());
    }//GEN-LAST:event_seleccionActionPerformed

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        learnArea = new Rectangle();
        areaQeue.add(learnArea);
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        areaDelete = true;
        areaSelect = false;
    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        cleanActionPerformed(evt);
    }//GEN-LAST:event_jButton3ActionPerformed

    private void jButton4ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton4ActionPerformed
        areaSelect = true;
        areaDelete = false;
    }//GEN-LAST:event_jButton4ActionPerformed

    private void cargaPreparadaActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cargaPreparadaActionPerformed
        String _filename2 = System.getProperty("user.dir") + testBaseUrl + (String)jComboBox1.getSelectedItem() + sufxType;
        log.info("directorio user <{}>",System.getProperty("user.dir"));
        
        try {
            destBuffImage = ImageIO.read(new File(_filename2));
            bufferImageFiltered = destBuffImage;
            
            java.awt.EventQueue.invokeLater(() -> {
                vista.repaint();
            });
        } catch (IOException ex) {
            log.error("error de archivo <{}>", _filename2, ex);
        }
    }//GEN-LAST:event_cargaPreparadaActionPerformed

    private void cargaOriginalActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cargaOriginalActionPerformed
        destBuffImage = buffImage;
        bufferImageFiltered = destBuffImage;
        java.awt.EventQueue.invokeLater(() -> {
            respuesta.repaint();
        });
        
    }//GEN-LAST:event_cargaOriginalActionPerformed

    public BufferedImage createCompatibleDestImage(BufferedImage src, ColorModel destCM) {
        BufferedImage image;

        int w = src.getWidth();
        int h = src.getHeight();

        WritableRaster wr = null;

        if (destCM == null) {
            destCM = src.getColorModel();
            // Not much support for ICM
            if (destCM instanceof IndexColorModel) {
                destCM = ColorModel.getRGBdefault();
            } else {
                /* Create destination image as similar to the source
                 *  as it possible...
                 */
                wr = src.getData().createCompatibleWritableRaster(w, h);
            }
        }

        if (wr == null) {
            /* This is the case when destination color model
             * was explicitly specified (and it may be not compatible
             * with source raster structure) or source is indexed image.
             * We should use destination color model to create compatible
             * destination raster here.
             */
            wr = destCM.createCompatibleWritableRaster(w, h);
        }

        image = new BufferedImage (destCM, wr,
                                   destCM.isAlphaPremultiplied(), null);

        return image;
    }
    
    private float calculaMatriz(int i, int j){
        float retorno;
        float half = (float)kwidth / 2;
        float di = (float) i - half;
        float dj = (float) j - half;
        
        retorno = 0.5f - (float) 1/ ( 1 + di * di + dj * dj);
//        double dist = Math.toRadians(   Math.sqrt((di*di+dj*dj) / (half*half*2)) * 90 ) ;
//        double dist = Math.toRadians(   Math.sqrt((dj*dj) / (half*half*2)) * 90 ) ;
//        if(dist == 0 )
//            retorno = -1; 
//        else
////            retorno = (float) (Math.c(dist)); 
////            retorno = (float) (Math.sin(dist)); 
//            retorno = -(float) (Math.sin(dist) / dist ) ; 
        
//        return  (1- retorno);
        return retorno;
    }
    
    private javax.swing.JPanel getNuevaVista(){
        return new JPanel(true){

            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                if(Objects.nonNull(buffImage)){
                    Graphics2D localg = (Graphics2D)g;
                    float escala = (float)vista.getBounds().width / (float)buffImage.getWidth();
                    
                    AffineTransform xforM = AffineTransform.getScaleInstance(escala, escala);
                    AffineTransformOp rop = new AffineTransformOp(xforM, AffineTransformOp.TYPE_BILINEAR);
                    localg.drawImage(buffImage, rop, 0     , 0);

                    areaQeue.forEach( a -> {
                        Rectangle evalArea = new Rectangle(a);
                        evalArea.x = (int) (evalArea.x * escala);
                        evalArea.y = (int) (evalArea.y * escala);
                        evalArea.width = (int) (evalArea.width * escala);
                        evalArea.height = (int) (evalArea.height * escala);
                        if (a.equals(learnArea)) {
                            localg.setColor(Color.RED);
                            
                        } else {
                            localg.setColor(Color.BLACK);
                            
                        }
                        localg.draw(evalArea);
                    
                    });
                    
                }
            }

            
        };
    }
    
    private javax.swing.JPanel getNuevaRespuesta(){
        return new JPanel(true){

            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                if(Objects.nonNull(bufferImageFiltered)){
                    Graphics2D localg = (Graphics2D)g;
                    float escala = (float)respuesta.getBounds().width / (float)bufferImageFiltered.getWidth();
                    AffineTransform xforM = AffineTransform.getScaleInstance(escala, escala);
                    AffineTransformOp rop = new AffineTransformOp(xforM, AffineTransformOp.TYPE_BILINEAR);
                    localg.drawImage(bufferImageFiltered, rop, 0     , 0);
                    
                }
            }

            
        };
    }
    
    private int compareTo(ParOrdenado i1, ParOrdenado i2){
        int compared = i1.getColumna().compareTo(i2.getColumna());
        return compared==0?i1.getFila().compareTo(i2.getFila()):compared;
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(VisLoad.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(VisLoad.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(VisLoad.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(VisLoad.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(() -> {
            new VisLoad().setVisible(true);
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JCheckBox actualizacion;
    private javax.swing.JButton cargaImagen;
    private javax.swing.JButton cargaOriginal;
    private javax.swing.JButton cargaPreparada;
    private javax.swing.JButton cargar;
    private javax.swing.JButton clean;
    private javax.swing.JButton enmascaraResultado;
    private javax.swing.JCheckBox entrenar;
    private javax.swing.JToggleButton freno;
    private javax.swing.JSpinner hiddNeurs;
    private javax.swing.JSpinner hiddenLearningRate;
    private javax.swing.JSpinner inNeurs;
    private javax.swing.JSpinner iteraciones;
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JButton jButton4;
    private javax.swing.JCheckBox jCheckBox1;
    private javax.swing.JComboBox jComboBox1;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JPanel jPanel3;
    private javax.swing.JPanel jPanel4;
    private javax.swing.JPanel jPanel5;
    private javax.swing.JPanel jPanel6;
    private javax.swing.JSeparator jSeparator1;
    private javax.swing.JSplitPane jSplitPane1;
    private javax.swing.JTabbedPane jTabbedPane1;
    private javax.swing.JSpinner outNeurs;
    private javax.swing.JSpinner outputLearningRate;
    private javax.swing.JButton procesar;
    private javax.swing.JPanel respuesta;
    private javax.swing.JButton salva;
    private javax.swing.JCheckBox seleccion;
    private javax.swing.JButton suavizaResultado;
    private javax.swing.JPanel vista;
    // End of variables declaration//GEN-END:variables
}
