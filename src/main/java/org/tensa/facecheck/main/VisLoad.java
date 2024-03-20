package org.tensa.facecheck.main;

import org.tensa.facecheck.weight.WeightModelingEnum;
import org.tensa.facecheck.weight.WeightCreationEnum;
import org.tensa.facecheck.activation.ActivationFunctionEnum;
import java.awt.CardLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ConvolveOp;
import java.awt.image.IndexColorModel;
import java.awt.image.Kernel;
import java.awt.image.WritableRaster;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.Vector;
import java.util.function.UnaryOperator;
import java.util.stream.IntStream;
import javax.imageio.ImageIO;
import javax.swing.AbstractCellEditor;
import javax.swing.ComboBoxModel;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JComboBox;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.JTable;
import javax.swing.SpinnerListModel;
import javax.swing.SpinnerModel;
import javax.swing.SpinnerNumberModel;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.activation.Activation;
import org.tensa.facecheck.activation.impl.ComposeActivationImpl;
import org.tensa.facecheck.activation.impl.IsMaxActivationImpl;
import org.tensa.facecheck.activation.impl.IsMinActivationImpl;
import org.tensa.facecheck.activation.impl.LeakyReluActivationImpl;
import org.tensa.facecheck.activation.impl.LinealActivationImpl;
import org.tensa.facecheck.activation.impl.NormalActivationImpl;
import org.tensa.facecheck.activation.impl.ReluActivationImpl;
import org.tensa.facecheck.activation.impl.SigmoidActivationImpl;
import org.tensa.facecheck.activation.impl.SoftMaxActivationImpl;
import org.tensa.facecheck.activation.impl.SoftPlusActivationImpl;
import org.tensa.facecheck.activation.impl.SoftSignActivationImpl;
import org.tensa.facecheck.activation.impl.StepActivationImpl;
import org.tensa.facecheck.activation.impl.TanHyperActivationImpl;
import org.tensa.facecheck.activation.impl.TheMaxActivationImpl;
import org.tensa.facecheck.activation.impl.TheMinActivationImpl;
import org.tensa.facecheck.filter.MaskOp;
import org.tensa.facecheck.layer.impl.OutputScale;
import org.tensa.facecheck.weight.WeightModelingStyle;
import org.tensa.facecheck.mapping.PixelMappings;
import org.tensa.facecheck.network.AbstractManager;
import org.tensa.facecheck.network.impl.ManagerBackPropImpl;
import org.tensa.facecheck.network.impl.BasicLearningEstrategyImpl;
import org.tensa.facecheck.network.impl.BasicLearningEstrategyEnum;
import org.tensa.facecheck.weight.WeightCreationStyle;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.FloatMatriz;
import org.tensa.tensada.matrix.NumericMatriz;
import org.tensa.tensada.matrix.ParOrdenado;
import org.tensa.facecheck.network.LearningEstrategy;

/**
 *
 * @author lorenzo
 */
public class VisLoad extends javax.swing.JFrame {

    private final Logger log = LoggerFactory.getLogger(VisLoad.class);

    private final String sufxType = ".jpg";
    private BufferedImage buffImage;
    private BufferedImage destBuffImage;
    private BufferedImage bufferImageFiltered;
    private Rectangle learnArea;
    private SeletionStatus areaStatus = SeletionStatus.MODIFY;
    private final FileNameExtensionFilter fileNameExtensionFilter = new FileNameExtensionFilter("Pesos Neuronas", "da3");
    private final FileNameExtensionFilter fileNameNetworkExtensionFilter = new FileNameExtensionFilter("Red Neuronas", "nn3");
    private final FileNameExtensionFilter fileNameExtensionFilterImage = new FileNameExtensionFilter("JPEG", "jpg");
    private final Rectangle leftTopPoint;
    private final Rectangle widthHwightpoint;

    private AbstractManager<Float> networkManager;

    public SpinnerModel getSpinnerModel() {
        return LearningFactorTableCellEditorImpl.getSpinnerModel();
    }

    public SpinnerModel getCuadradoSpinnerModel() {
        return NeuronTableCellEditorImpl.getCuadradoSpinnerModel();
    }
    
    public ComboBoxModel getBufferComboboxModel() {
        return new DefaultComboBoxModel<>(new Integer[]{ 1, 3, 6, 12, 15, 18, 21});
    }

    /**
     * Get the value of sufxType
     *
     * @return the value of sufxType
     */
    public String getSufxType() {
        return sufxType;
    }

    public String getRawPixelViewSizeFromInput(Integer value) {
        int raw = (int) Math.sqrt(value / 3);
        return String.format(" %d x %d X 3", raw, raw);
    }

    public String getRawPixelInNeur() {
        return getRawPixelViewSizeFromInput((Integer) inNeurs.getValue());
    }

    public String getRawPixelOutNeur() {
        int rowCount = jTableWeight.getRowCount();
        Object value = jTableWeight.getValueAt(rowCount - 1, 0);
        return getRawPixelViewSizeFromInput((Integer) value);
    }

    /**
     * Creates new form visLoad
     */
    public VisLoad() {
        initComponents();
        learnArea = new Rectangle();
        learnArea.setSize(100, 100);
        learnArea.setLocation(10, 10);
        leftTopPoint = new Rectangle();
        widthHwightpoint = new Rectangle();
        networkManager = new ManagerBackPropImpl<>();
        networkManager.setSupplier((Dominio dominio) -> new FloatMatriz(dominio));
        networkManager.setPixelMapper(PixelMappings.defaultMapping());
        networkManager.getAreaQeue().add(learnArea);
        jTableWeight.getModel().setValueAt(WeightCreationEnum.RANDOM, 0, 2);
        jTableWeight.getModel().setValueAt(WeightCreationEnum.RANDOM, 1, 2);
        jTableWeight.getModel().setValueAt(WeightModelingEnum.NORMALIZED, 0, 3);
        jTableWeight.getModel().setValueAt(WeightModelingEnum.NORMALIZED, 1, 3);
        jTableWeight.getModel().setValueAt(ActivationFunctionEnum.LINEAL, 0, 4);
        jTableWeight.getModel().setValueAt(ActivationFunctionEnum.LINEAL, 1, 4);
        jTableWeight.getModel().setValueAt(BasicLearningEstrategyEnum.TREE_ADV_ONE, 0, 6);
        jTableWeight.getModel().setValueAt(BasicLearningEstrategyEnum.ONE_ADV_ONE_TREE_BACK_ONE, 1, 6);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        adaptInputButtonGroup = new javax.swing.ButtonGroup();
        jFileChooserImagenSalva = new javax.swing.JFileChooser();
        jFileChooserLoadImagenResult = new javax.swing.JFileChooser();
        jFileChooserLoadImagen = new javax.swing.JFileChooser();
        jFileChooserSaveNet = new javax.swing.JFileChooser();
        jFileChooserLoadNet = new javax.swing.JFileChooser();
        jPanelTop = new javax.swing.JPanel();
        jTabbedPane1 = new javax.swing.JTabbedPane();
        jPanel1 = new javax.swing.JPanel();
        cargaImagen = new javax.swing.JButton();
        scaleJRadioButton = new javax.swing.JRadioButton();
        normalizeJRadioButton = new javax.swing.JRadioButton();
        reflectJRadioButton = new javax.swing.JRadioButton();
        preventJRadioButton = new javax.swing.JRadioButton();
        libreJRadioButton = new javax.swing.JRadioButton();
        jCheckBoxScale1neg1 = new javax.swing.JCheckBox();
        normaExtJRadioButton = new javax.swing.JRadioButton();
        jPanel2 = new javax.swing.JPanel();
        suavizaResultado = new javax.swing.JButton();
        enmascaraResultado = new javax.swing.JButton();
        cargaOriginal = new javax.swing.JButton();
        cargaPreparada = new javax.swing.JButton();
        jSeparator1 = new javax.swing.JSeparator();
        jSeparator2 = new javax.swing.JSeparator();
        jButtonSalvaImagen = new javax.swing.JButton();
        jPanel3 = new javax.swing.JPanel();
        clean = new javax.swing.JButton();
        jLabel1 = new javax.swing.JLabel();
        inNeurs = new javax.swing.JSpinner();
        jLabel2 = new javax.swing.JLabel();
        jButtonAddRow = new javax.swing.JButton();
        jButtonRemoveRow = new javax.swing.JButton();
        jLabelNumInPixels = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        jLabelNumOutPixels = new javax.swing.JLabel();
        jButtonSalvaRed = new javax.swing.JButton();
        jButtonCargaRed = new javax.swing.JButton();
        jPanel4 = new javax.swing.JPanel();
        procesar = new javax.swing.JButton();
        entrenar = new javax.swing.JCheckBox();
        iteraciones = new javax.swing.JSpinner();
        seleccionCopy = new javax.swing.JCheckBox();
        cleanCopy = new javax.swing.JButton();
        freno = new javax.swing.JToggleButton();
        actualizacion = new javax.swing.JCheckBox();
        jCheckBoxParalell = new javax.swing.JCheckBox();
        jComboBoxBufferSize = new javax.swing.JComboBox<>();
        jPanel5 = new javax.swing.JPanel();
        seleccion = new javax.swing.JCheckBox();
        addSelectionButton = new javax.swing.JButton();
        deleteSelectionButton = new javax.swing.JButton();
        duplicaSelectionButton = new javax.swing.JButton();
        limpiajButton = new javax.swing.JButton();
        jErrorGraf = getNuevaErrorGram();
        jPanelCard = new javax.swing.JPanel();
        jSplitPane1 = new javax.swing.JSplitPane();
        vista = getNuevaVista();
        respuesta = getNuevaRespuesta();
        jScrollPane1 = new javax.swing.JScrollPane();
        jTableWeight = new javax.swing.JTable();
        jProgressBar1 = new javax.swing.JProgressBar();

        jFileChooserImagenSalva.setDialogType(javax.swing.JFileChooser.SAVE_DIALOG);
        jFileChooserImagenSalva.setCurrentDirectory(new File(System.getProperty("user.dir")));
        jFileChooserImagenSalva.setDialogTitle("Salva Imagen");
        jFileChooserImagenSalva.setFileFilter(getFileNameExtensionFilterImage());

        jFileChooserLoadImagenResult.setCurrentDirectory(new File(System.getProperty("user.dir")));
        jFileChooserLoadImagenResult.setDialogTitle("Carga resultante");
        jFileChooserLoadImagenResult.setFileFilter(getFileNameExtensionFilterImage());

        jFileChooserLoadImagen.setCurrentDirectory(new File(System.getProperty("user.dir")));
        jFileChooserLoadImagen.setDialogTitle("Carga Imagen Inicial");
        jFileChooserLoadImagen.setFileFilter(getFileNameExtensionFilterImage());

        jFileChooserSaveNet.setDialogType(javax.swing.JFileChooser.SAVE_DIALOG);
        jFileChooserSaveNet.setCurrentDirectory(new File(System.getProperty("user.dir")));
        jFileChooserSaveNet.setDialogTitle("Salva Red");
        jFileChooserSaveNet.setFileFilter(getFileNameNetworkExtensionFilter());

        jFileChooserLoadNet.setCurrentDirectory(new File(System.getProperty("user.dir")));
        jFileChooserLoadNet.setDialogTitle("Carga Red");
        jFileChooserLoadNet.setFileFilter(getFileNameNetworkExtensionFilter());

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Vista de carga");

        cargaImagen.setText("Cargar...");
        cargaImagen.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cargaImagenActionPerformed(evt);
            }
        });

        adaptInputButtonGroup.add(scaleJRadioButton);
        scaleJRadioButton.setText("Escalar");

        adaptInputButtonGroup.add(normalizeJRadioButton);
        normalizeJRadioButton.setSelected(true);
        normalizeJRadioButton.setText("Normalizar");

        adaptInputButtonGroup.add(reflectJRadioButton);
        reflectJRadioButton.setText("Reflectancia");

        adaptInputButtonGroup.add(preventJRadioButton);
        preventJRadioButton.setText("Previene ceros");

        adaptInputButtonGroup.add(libreJRadioButton);
        libreJRadioButton.setText("Libre");

        jCheckBoxScale1neg1.setText("-1 +1");
        jCheckBoxScale1neg1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jCheckBoxScale1neg1ActionPerformed(evt);
            }
        });

        adaptInputButtonGroup.add(normaExtJRadioButton);
        normaExtJRadioButton.setText("Norma-ext");

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(cargaImagen)
                .addGap(18, 18, 18)
                .addComponent(scaleJRadioButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(normalizeJRadioButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(normaExtJRadioButton, javax.swing.GroupLayout.PREFERRED_SIZE, 98, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(reflectJRadioButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(preventJRadioButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(libreJRadioButton)
                .addGap(136, 136, 136)
                .addComponent(jCheckBoxScale1neg1)
                .addContainerGap())
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cargaImagen)
                    .addComponent(scaleJRadioButton)
                    .addComponent(normalizeJRadioButton)
                    .addComponent(reflectJRadioButton)
                    .addComponent(preventJRadioButton)
                    .addComponent(libreJRadioButton)
                    .addComponent(jCheckBoxScale1neg1)
                    .addComponent(normaExtJRadioButton))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        jTabbedPane1.addTab("Imagen entrada", null, jPanel1, "Carga de imagen");

        suavizaResultado.setText("Suavizar");
        suavizaResultado.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                suavizaResultadoActionPerformed(evt);
            }
        });

        enmascaraResultado.setText("enmascarar");
        enmascaraResultado.setToolTipText("mascara de salida preparada sobre original");
        enmascaraResultado.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                enmascaraResultadoActionPerformed(evt);
            }
        });

        cargaOriginal.setText("Limpiar original");
        cargaOriginal.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cargaOriginalActionPerformed(evt);
            }
        });

        cargaPreparada.setText("Cargar...");
        cargaPreparada.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cargaPreparadaActionPerformed(evt);
            }
        });

        jSeparator1.setOrientation(javax.swing.SwingConstants.VERTICAL);

        jSeparator2.setOrientation(javax.swing.SwingConstants.VERTICAL);

        jButtonSalvaImagen.setText("Salvar...");
        jButtonSalvaImagen.setToolTipText("");
        jButtonSalvaImagen.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonSalvaImagenActionPerformed(evt);
            }
        });

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
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jSeparator2, javax.swing.GroupLayout.PREFERRED_SIZE, 17, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jButtonSalvaImagen)
                .addContainerGap(379, Short.MAX_VALUE))
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
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jSeparator1)
                    .addComponent(jSeparator2))
                .addContainerGap())
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel2Layout.createSequentialGroup()
                .addGap(0, 0, Short.MAX_VALUE)
                .addComponent(jButtonSalvaImagen))
        );

        jTabbedPane1.addTab("Imagen salida", null, jPanel2, "Pre proceso de imagen de salida");

        jPanel3.addComponentListener(new java.awt.event.ComponentAdapter() {
            public void componentHidden(java.awt.event.ComponentEvent evt) {
                jPanel3ComponentHidden(evt);
            }
            public void componentShown(java.awt.event.ComponentEvent evt) {
                jPanel3ComponentShown(evt);
            }
        });

        clean.setText("Limpiar");
        clean.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cleanActionPerformed(evt);
            }
        });

        jLabel1.setText("Entrada");

        inNeurs.setModel(getCuadradoSpinnerModel());
        inNeurs.setToolTipText("Neuronas de entrada (pixels)");
        inNeurs.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                inNeursStateChanged(evt);
            }
        });
        inNeurs.addComponentListener(new java.awt.event.ComponentAdapter() {
            public void componentShown(java.awt.event.ComponentEvent evt) {
                inNeursComponentShown(evt);
            }
        });

        jLabel2.setText("Pixels");

        jButtonAddRow.setText("+");
        jButtonAddRow.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonAddRowActionPerformed(evt);
            }
        });

        jButtonRemoveRow.setLabel("-");
        jButtonRemoveRow.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonRemoveRowActionPerformed(evt);
            }
        });

        jLabelNumInPixels.setText("0/3");

        jLabel3.setText("Salida");

        jLabelNumOutPixels.setText("0/3");

        jButtonSalvaRed.setText("Salva Red");
        jButtonSalvaRed.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonSalvaRedActionPerformed(evt);
            }
        });

        jButtonCargaRed.setText("Carga Red");
        jButtonCargaRed.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonCargaRedActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel3Layout = new javax.swing.GroupLayout(jPanel3);
        jPanel3.setLayout(jPanel3Layout);
        jPanel3Layout.setHorizontalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(clean)
                .addGap(18, 18, 18)
                .addComponent(jLabel1)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(inNeurs, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(24, 24, 24)
                .addComponent(jLabel2)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabelNumInPixels, javax.swing.GroupLayout.PREFERRED_SIZE, 76, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(jLabel3)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabelNumOutPixels)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 246, Short.MAX_VALUE)
                .addComponent(jButtonCargaRed)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jButtonSalvaRed)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jButtonAddRow)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jButtonRemoveRow)
                .addContainerGap())
        );
        jPanel3Layout.setVerticalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(clean)
                    .addComponent(jLabel1)
                    .addComponent(inNeurs, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel2)
                    .addComponent(jButtonAddRow)
                    .addComponent(jButtonRemoveRow)
                    .addComponent(jLabel3)
                    .addComponent(jLabelNumOutPixels)
                    .addComponent(jButtonSalvaRed)
                    .addComponent(jButtonCargaRed)
                    .addComponent(jLabelNumInPixels, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addGap(0, 0, Short.MAX_VALUE))
        );

        jTabbedPane1.addTab("Capas/Pesos", null, jPanel3, "administracion de pesos");

        procesar.setText("Procesar");
        procesar.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                procesarActionPerformed(evt);
            }
        });

        entrenar.setSelected(true);
        entrenar.setText("entrenar");
        entrenar.setToolTipText("modo de proceso");

        iteraciones.setModel(new javax.swing.SpinnerNumberModel(50, 1, 500, 10));
        iteraciones.setToolTipText("Iteraciones");
        iteraciones.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                iteracionesStateChanged(evt);
            }
        });

        seleccionCopy.setText("Usa Selección");
        seleccionCopy.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                seleccionCopyActionPerformed(evt);
            }
        });

        cleanCopy.setText("Limpiar");
        cleanCopy.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cleanCopyActionPerformed(evt);
            }
        });

        freno.setText("Freno");

        actualizacion.setText("Actualización");
        actualizacion.setToolTipText("actualizacion de pantalla cada 30 segundos");

        jCheckBoxParalell.setSelected(true);
        jCheckBoxParalell.setText("proceso paralelo");

        jComboBoxBufferSize.setModel(getBufferComboboxModel());
        jComboBoxBufferSize.setSelectedIndex(0);
        jComboBoxBufferSize.setSelectedItem(0);
        jComboBoxBufferSize.setToolTipText("tamaño buffer");

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
                .addComponent(jCheckBoxParalell)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jComboBoxBufferSize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(iteraciones, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(30, 30, 30)
                .addComponent(seleccionCopy)
                .addGap(18, 18, 18)
                .addComponent(cleanCopy)
                .addGap(18, 18, 18)
                .addComponent(actualizacion)
                .addContainerGap(57, Short.MAX_VALUE))
        );
        jPanel4Layout.setVerticalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel4Layout.createSequentialGroup()
                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(entrenar)
                    .addComponent(procesar)
                    .addComponent(iteraciones, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(seleccionCopy)
                    .addComponent(cleanCopy)
                    .addComponent(freno)
                    .addComponent(actualizacion)
                    .addComponent(jCheckBoxParalell)
                    .addComponent(jComboBoxBufferSize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
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

        addSelectionButton.setText("Agrega selección");
        addSelectionButton.setEnabled(seleccion.isSelected());
        addSelectionButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                addSelectionButtonActionPerformed(evt);
            }
        });

        deleteSelectionButton.setText("Quita selección");
        deleteSelectionButton.setEnabled(seleccion.isSelected());
        deleteSelectionButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                deleteSelectionButtonActionPerformed(evt);
            }
        });

        duplicaSelectionButton.setText("Duplicar selección");
        duplicaSelectionButton.setEnabled(seleccion.isSelected());
        duplicaSelectionButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                duplicaSelectionButtonActionPerformed(evt);
            }
        });

        limpiajButton.setText("Limpia");
        limpiajButton.setEnabled(seleccion.isSelected());
        limpiajButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                limpiajButtonActionPerformed(evt);
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
                .addComponent(addSelectionButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(duplicaSelectionButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(deleteSelectionButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(limpiajButton)
                .addContainerGap(336, Short.MAX_VALUE))
        );
        jPanel5Layout.setVerticalGroup(
            jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel5Layout.createSequentialGroup()
                .addGroup(jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(seleccion)
                    .addComponent(addSelectionButton)
                    .addComponent(deleteSelectionButton)
                    .addComponent(duplicaSelectionButton)
                    .addComponent(limpiajButton))
                .addGap(0, 0, Short.MAX_VALUE))
        );

        jTabbedPane1.addTab("Seleccion", jPanel5);

        javax.swing.GroupLayout jPanelTopLayout = new javax.swing.GroupLayout(jPanelTop);
        jPanelTop.setLayout(jPanelTopLayout);
        jPanelTopLayout.setHorizontalGroup(
            jPanelTopLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jTabbedPane1)
        );
        jPanelTopLayout.setVerticalGroup(
            jPanelTopLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanelTopLayout.createSequentialGroup()
                .addComponent(jTabbedPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 57, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 6, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout jErrorGrafLayout = new javax.swing.GroupLayout(jErrorGraf);
        jErrorGraf.setLayout(jErrorGrafLayout);
        jErrorGrafLayout.setHorizontalGroup(
            jErrorGrafLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );
        jErrorGrafLayout.setVerticalGroup(
            jErrorGrafLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 14, Short.MAX_VALUE)
        );

        jPanelCard.setLayout(new java.awt.CardLayout());

        jSplitPane1.setDividerLocation(128);

        vista.setBorder(javax.swing.BorderFactory.createEmptyBorder(1, 1, 1, 1));
        vista.setCursor(new java.awt.Cursor(java.awt.Cursor.DEFAULT_CURSOR));
        vista.addMouseMotionListener(new java.awt.event.MouseMotionAdapter() {
            public void mouseDragged(java.awt.event.MouseEvent evt) {
                vistaMouseDragged(evt);
            }
            public void mouseMoved(java.awt.event.MouseEvent evt) {
                vistaMouseMoved(evt);
            }
        });
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
            .addGap(0, 452, Short.MAX_VALUE)
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
            .addGap(0, 452, Short.MAX_VALUE)
        );

        jSplitPane1.setRightComponent(respuesta);

        jPanelCard.add(jSplitPane1, "showCard");

        jTableWeight.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {
                { new Integer(15), null, "RANDOM", "NORMALIZED", "LINEAL",  new Float(5.0E-5), "ONE_ADV_ONE_TREE_BACK_ONE"},
                { new Integer(27), null, "RANDOM", "NORMALIZED", "LINEAL",  new Float(5.0E-5), "ONE_ADV_ONE_TREE_BACK_ONE"}
            },
            new String [] {
                "Neuronas", "Tendencia", "Creacion Pesos", "Estilo Pesos", "Func. Activacion", "Fact. Aprendisaje", "estratg. Aprendisaje"
            }
        ) {
            Class[] types = new Class [] {
                java.lang.Integer.class, java.lang.Boolean.class, java.lang.Object.class, java.lang.Object.class, java.lang.Object.class, java.lang.Float.class, java.lang.Object.class
            };

            public Class getColumnClass(int columnIndex) {
                return types [columnIndex];
            }
        });
        jTableWeight.setColumnSelectionAllowed(true);
        jTableWeight.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_SELECTION);
        jTableWeight.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_SELECTION);
        jTableWeight.getTableHeader().setReorderingAllowed(false);
        jTableWeight.addPropertyChangeListener(new java.beans.PropertyChangeListener() {
            public void propertyChange(java.beans.PropertyChangeEvent evt) {
                jTableWeightPropertyChange(evt);
            }
        });
        jScrollPane1.setViewportView(jTableWeight);
        jTableWeight.getColumnModel().getSelectionModel().setSelectionMode(javax.swing.ListSelectionModel.SINGLE_SELECTION);
        if (jTableWeight.getColumnModel().getColumnCount() > 0) {
            jTableWeight.getColumnModel().getColumn(0).setResizable(false);
            jTableWeight.getColumnModel().getColumn(0).setCellEditor(getNeuronCellEditor());
            jTableWeight.getColumnModel().getColumn(1).setResizable(false);
            jTableWeight.getColumnModel().getColumn(2).setResizable(false);
            jTableWeight.getColumnModel().getColumn(2).setCellEditor(getNeuronCreationWeigth());
            jTableWeight.getColumnModel().getColumn(3).setResizable(false);
            jTableWeight.getColumnModel().getColumn(3).setCellEditor(getNeuronStyleWeigth());
            jTableWeight.getColumnModel().getColumn(4).setResizable(false);
            jTableWeight.getColumnModel().getColumn(4).setCellEditor(getActivationFunctionCellEditor());
            jTableWeight.getColumnModel().getColumn(5).setResizable(false);
            jTableWeight.getColumnModel().getColumn(5).setCellEditor(getLearningFactorCellEditor());
            jTableWeight.getColumnModel().getColumn(5).setCellRenderer(getLearningFactorCellRender());
            jTableWeight.getColumnModel().getColumn(6).setResizable(false);
            jTableWeight.getColumnModel().getColumn(6).setCellEditor(getLearningEstrategyCellEditor());
        }

        jPanelCard.add(jScrollPane1, "cardPesos");

        jProgressBar1.setStringPainted(true);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jPanelTop, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jProgressBar1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jErrorGraf, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                .addContainerGap())
            .addComponent(jPanelCard, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanelTop, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jPanelCard, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(jErrorGraf, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jProgressBar1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );

        getAccessibleContext().setAccessibleName("Oculta Caras");

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void suavizaResultadoActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_suavizaResultadoActionPerformed
        final int kwidth = 27;
        float[] data = new float[kwidth * kwidth];
        float total = 0;

        for (int i = 0; i < kwidth * kwidth; i++) {
            data[i] = calculaMatriz(i % kwidth, i / kwidth, kwidth);
            total += data[i];

        }

        for (int i = 0; i < kwidth * kwidth; i++) {
            data[i] /= total / 1.5;
        }
        
        ConvolveOp conv = new ConvolveOp(new Kernel(kwidth, kwidth, data));
        if (Objects.nonNull(bufferImageFiltered)) {
            bufferImageFiltered.flush();
        }

        bufferImageFiltered = conv.filter(destBuffImage, null);
        destBuffImage = bufferImageFiltered;

        java.awt.EventQueue.invokeLater(() -> {
            respuesta.repaint();
        });
    }//GEN-LAST:event_suavizaResultadoActionPerformed

    private void enmascaraResultadoActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_enmascaraResultadoActionPerformed

        MaskOp conv = new MaskOp();
        conv.setOtherSrc(destBuffImage);
        if (Objects.nonNull(bufferImageFiltered)) {
            bufferImageFiltered.flush();
        }

        bufferImageFiltered = conv.filter(buffImage, null);
        destBuffImage = bufferImageFiltered;

        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
            respuesta.repaint();
        });
    }//GEN-LAST:event_enmascaraResultadoActionPerformed

    @SuppressWarnings("unchecked")
    private void procesarActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_procesarActionPerformed

        disableConsole();
        
        bufferImageFiltered = createCompatibleDestImage(buffImage, null);

        networkManager.setInputImage(buffImage);
        networkManager.setOutputImage(bufferImageFiltered);
        networkManager.setCompareImage(destBuffImage);
        networkManager.setTrainingMode(entrenar.isSelected());
        networkManager.setBaseParallel(jCheckBoxParalell.isSelected());
        networkManager.setSlotBuffer((Integer)jComboBoxBufferSize.getSelectedItem());

        networkManager.setLearningRate(IntStream.range(0, jTableWeight.getRowCount())
                .mapToObj(i -> (Float) jTableWeight.getValueAt(i, 5)).toArray(Float[]::new));

        LearningEstrategy<Float>[] learningControl = IntStream.range(0, jTableWeight.getRowCount())
                .mapToObj(i -> {
                    BasicLearningEstrategyEnum ble = (BasicLearningEstrategyEnum) jTableWeight.getValueAt(i, 6);
                    UnaryOperator<Integer> learninEstrategy2Control = learninEstrategy2Control(ble);
                    return new BasicLearningEstrategyImpl<Float>(learninEstrategy2Control, LearningEstrategy.floatBasicLearningSeries);
                })
                .toArray(BasicLearningEstrategyImpl[]::new);
        networkManager.setLearningControl(learningControl);
        Activation<Float>[] activationFunction
                = IntStream.range(0, jTableWeight.getRowCount())
                        .mapToObj(i -> (ActivationFunctionEnum) jTableWeight.getValueAt(i, 4))
                        .map(this::activation2Activation)
                        .toArray(Activation[]::new);
        networkManager.setActivationFunction(activationFunction);
        
        Boolean[] useBias = IntStream.range(0, jTableWeight.getRowCount())
                .mapToObj(i -> (Boolean) jTableWeight.getValueAt(i, 1))
                .map(b -> Objects.isNull(b)?Boolean.FALSE:b)
                .peek(c -> log.info("use bias <{}>", c))
                .toArray(Boolean[]::new);
        networkManager.setUseBias(useBias);
        networkManager.setBaseParallel(jCheckBoxParalell.isSelected());

        networkManager.setIterateTo((int) iteraciones.getValue());
        networkManager.setUseSelection(seleccion.isSelected());

        networkManager.setPixelMapper(PixelMappings.defaultMapping());
        if (!jCheckBoxScale1neg1.isSelected()) {

            if (scaleJRadioButton.isSelected()) {
                networkManager.setInputScale(OutputScale::scale);
            } else if (normalizeJRadioButton.isSelected()) {
                networkManager.setInputScale(OutputScale::normalized);
            } else if (normaExtJRadioButton.isSelected()) {
                networkManager.setInputScale(OutputScale::normalizedPus1);
                networkManager.setPixelMapper(PixelMappings.plus1Mapping());
            } else if (reflectJRadioButton.isSelected()) {
                networkManager.setInputScale(OutputScale::reflectance);
            } else if (preventJRadioButton.isSelected()) {
                networkManager.setInputScale(OutputScale::prevent01);
            } else {
                networkManager.setInputScale(OutputScale::sameEscale);
            }

        } else {

            if (scaleJRadioButton.isSelected()) {
                networkManager.setInputScale(OutputScale::extendedScale);
            } else if (normalizeJRadioButton.isSelected()) {
                networkManager.setInputScale(OutputScale::extendedNormalized);
            } else if (normalizeJRadioButton.isSelected()) {
                networkManager.setInputScale(OutputScale::extendedNormalizedPlus1);
                networkManager.setPixelMapper(PixelMappings.plus1Mapping());
            } else if (reflectJRadioButton.isSelected()) {
                networkManager.setInputScale(OutputScale::extendedReflectance);
            }
        }

        new Thread(() -> {
            try {
                networkManager.process();
                bufferImageFiltered = networkManager.getOutputImage();

            } catch (NullPointerException ex) {
                log.error("error al ejecutar red", ex);
                javax.swing.JOptionPane.showMessageDialog(null, "Matriz con parametros nulos", "Error", javax.swing.JOptionPane.ERROR_MESSAGE);                

            }

            enableConsole();

            java.awt.EventQueue.invokeLater(() -> {
                jProgressBar1.setValue(jProgressBar1.getMaximum());
                respuesta.repaint();
            });
        }).start();

        new Thread(() -> {
            while (!procesar.isEnabled()) {
                try {
                    Thread.sleep(5000);

                    networkManager.setEmergencyBreak(freno.isSelected());

                    IntStream.range(0, jTableWeight.getRowCount()).forEach(i -> {
                        Float o = networkManager.getLearningRate(i);
                        jTableWeight.setValueAt(o, i, 5);
                    });

                    networkManager.setIterateTo((int) iteraciones.getValue());
                    networkManager.setUseSelection(seleccion.isSelected());

                    jProgressBar1.setValue(networkManager.getIterateCurrent());
                    if (actualizacion.isSelected()) {

                        bufferImageFiltered = networkManager.getOutputImage();
                        respuesta.repaint();
                        jErrorGraf.repaint();
                        log.info("realiza actualizacion");

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
    
    @SuppressWarnings("unchecked")
    private void cleanActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cleanActionPerformed
        log.info("iniciando 0...");
        int inStep = (int) inNeurs.getValue();

        networkManager.setInputImage(buffImage);
        networkManager.setOutputImage(bufferImageFiltered);
        networkManager.setCompareImage(destBuffImage);

        networkManager.setInStep(inStep);
        networkManager.setHiddenStep(IntStream.range(0, jTableWeight.getRowCount())
                .map(i -> (Integer) jTableWeight.getValueAt(i, 0)).toArray());

        UnaryOperator<NumericMatriz<Float>>[] weightCreationStyle = IntStream.range(0, jTableWeight.getRowCount())
                .mapToObj(i -> (WeightCreationEnum) jTableWeight.getValueAt(i, 2))
                .peek(c -> log.info("creacion style <{}>", c))
                .map(this::creation2style)
                .toArray(UnaryOperator[]::new);

        UnaryOperator<NumericMatriz<Float>>[] weightModelingStyle = IntStream.range(0, jTableWeight.getRowCount())
                .mapToObj(i -> (WeightModelingEnum) jTableWeight.getValueAt(i, 3))
                .peek(m -> log.info("modelado style <{}>", m))
                .map(this::modeling2style)
                .toArray(UnaryOperator[]::new);
        
        Boolean[] useBias = IntStream.range(0, jTableWeight.getRowCount())
                .mapToObj(i -> (Boolean) jTableWeight.getValueAt(i, 1))
                .map(b -> Objects.isNull(b)?Boolean.FALSE:b)
                .peek(c -> log.info("use bias <{}>", c))
                .toArray(Boolean[]::new);
        
        networkManager.setUseBias(useBias);

        networkManager.initMatrix(weightCreationStyle, weightModelingStyle);

    }//GEN-LAST:event_cleanActionPerformed

    private Activation<Float> activation2Activation(ActivationFunctionEnum afe) {
        switch (afe) {
            case LINEAL:
                return new LinealActivationImpl<>();
            case NORMAL:
                return new NormalActivationImpl<>();
            case RELU:
                return new ReluActivationImpl<>();
            case LEAKYRELU_00_800:
                return new LeakyReluActivationImpl<>(.8f);
            case LEAKYRELU_00_500:
                return new LeakyReluActivationImpl<>(.5f);
            case LEAKYRELU_00_100:
                return new LeakyReluActivationImpl<>(.1f);
            case LEAKYRELU_00_080:
                return new LeakyReluActivationImpl<>(.08f);
            case LEAKYRELU_00_050:
                return new LeakyReluActivationImpl<>(.05f);
            case LEAKYRELU_00_010:
                return new LeakyReluActivationImpl<>(.01f);
            case LEAKYRELU_00_008:
                return new LeakyReluActivationImpl<>(.008f);
            case LEAKYRELU_00_005:
                return new LeakyReluActivationImpl<>(.005f);
            case LEAKYRELU_00_001:
                return new LeakyReluActivationImpl<>(.001f);
            case SIGMOIDE_00_1:
                return new SigmoidActivationImpl<>(.1f);
            case SIGMOIDE_00_5:
                return new SigmoidActivationImpl<>(.5f);
            case SIGMOIDE_00_8:
                return new SigmoidActivationImpl<>(.8f);
            case SIGMOIDE_01_0:
                return new SigmoidActivationImpl<>(1f);
            case SIGMOIDE_02_00:
                return new SigmoidActivationImpl<>(2f);
            case SIGMOIDE_04_00:
                return new SigmoidActivationImpl<>(4f);
            case SIGMOIDE_08_00:
                return new SigmoidActivationImpl<>(8f);
            case SIGMOIDE_16_00:
                return new SigmoidActivationImpl<>(16f);
            case SIGMOIDE_32_00:
                return new SigmoidActivationImpl<>(32f);
            case SIGMOIDE_64_00:
                return new SigmoidActivationImpl<>(64f);
            case SOFT_PLUS:
                return new SoftPlusActivationImpl<>();
            case SOFT_SIGN:
                return new SoftSignActivationImpl<>();
            case SOFT_MAX:
                return new SoftMaxActivationImpl<>();
            case TAN_HYPER:
                return new TanHyperActivationImpl<>();
            case IS_MIN:
                return new IsMinActivationImpl<>();
            case IS_MAX:
                return new IsMaxActivationImpl<>();
            case THE_MIN:
                return new TheMinActivationImpl<>();
            case THE_MAX:
                return new TheMaxActivationImpl<>();
            case THE_MIN_THYP:
                return new ComposeActivationImpl<>(new TheMinActivationImpl<>(),new TanHyperActivationImpl<>());
            case THE_MAX_THYP:
                return new ComposeActivationImpl<>(new TheMaxActivationImpl<>(),new TanHyperActivationImpl<>());
            case STEP:
            default:
                return new StepActivationImpl<>();
        }
    }

    private UnaryOperator<Integer> learninEstrategy2Control(BasicLearningEstrategyEnum blee) {
        switch (blee) {
            case ONE_ADV_ONE_TREE_BACK_ONE:
                return BasicLearningEstrategyImpl::cadaUnoAvanzaUnoCadaTresVuelvaUno;
            case TREE_ADV_ONE:
                return BasicLearningEstrategyImpl::cadaTresAvanzaUno;
            case TREE_ADV_ONE_NINE_BACK_ONE:
                return BasicLearningEstrategyImpl::cadaTresAvanzaUnoCadaNueveVuelvaUno;
            case NINE_ADV_ONE:
            default:
                return BasicLearningEstrategyImpl::cadaNueveAvanzaUno;
        }
    }

    private UnaryOperator<NumericMatriz<Float>> modeling2style(WeightModelingEnum wme) {
        switch (wme) {
            case SIMPLE:
                return WeightModelingStyle::simpleModelingStyle;
            case REFELCTANCE:
                return WeightModelingStyle::reflectanceModelingStyle;
            case NORMALIZED:
            default:
                return WeightModelingStyle::normalizedModelingStyle;
        }
    }

    private UnaryOperator<NumericMatriz<Float>> creation2style(WeightCreationEnum wce) {
        switch (wce) {
            case RANDOM:
                return WeightCreationStyle::randomCreationStyle;
            case DIAGONAL:
                return WeightCreationStyle::diagonalCreationStyle;
            case FROM_OUTSTART:
                return WeightCreationStyle::fromOutStarCreationStyle;
            case RANDOM_SEGMENTED:
            default:
                return WeightCreationStyle::randomSegmetedCreationStyle;
        }
    }

    private final TableCellEditor learningEstrategyCellEditor = new LearningEstrategyTableCellEditorImpl();

    public TableCellEditor getLearningEstrategyCellEditor() {
        return learningEstrategyCellEditor;
    }

    private final TableCellRenderer learningFactorCellRender = new LearningFactorCellRenderImp();

    public TableCellRenderer getLearningFactorCellRender() {
        return learningFactorCellRender;
    }

    private final TableCellEditor activationFunctionCellEditor = new ActivationFunctionTableCellEditorImpl();

    public TableCellEditor getActivationFunctionCellEditor() {
        return activationFunctionCellEditor;
    }

    private final TableCellEditor learningFactorCellEditor = new LearningFactorTableCellEditorImpl();

    public TableCellEditor getLearningFactorCellEditor() {
        return learningFactorCellEditor;
    }

    private final TableCellEditor neuronCreationWeigth = new CreationWeightTableCellEditorImpl();

    public TableCellEditor getNeuronCreationWeigth() {
        return neuronCreationWeigth;
    }

    private final TableCellEditor neuronStyleWeigth = new NeuronStyleTableCellEditorImpl();

    public TableCellEditor getNeuronStyleWeigth() {
        return neuronStyleWeigth;
    }

    private final TableCellEditor neuronCellEditor = new NeuronTableCellEditorImpl();

    public TableCellEditor getNeuronCellEditor() {
        return neuronCellEditor;
    }

    private void vistaMouseReleased(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_vistaMouseReleased
        int x = evt.getX();
        int y = evt.getY();
        if (Objects.isNull(buffImage)) {
            JOptionPane.showMessageDialog(null, "Imagen no seleccionada");
            return;
        }
        float escala = (float) buffImage.getWidth() / (float) vista.getBounds().width;

        switch (areaStatus) {
            case MODIFY_SIZE:

                learnArea.width = (int) (x * escala) - learnArea.x;
                learnArea.height = (int) (y * escala) - learnArea.y;
                areaStatus = SeletionStatus.MODIFY;

                java.awt.EventQueue.invokeLater(() -> {
                    vista.repaint();
                });
                break;
            case ADD:
            case MODIFY_POSITION:
                learnArea.x = (int) (x * escala);
                learnArea.y = (int) (y * escala);
                areaStatus = SeletionStatus.MODIFY;

                java.awt.EventQueue.invokeLater(() -> {
                    vista.repaint();
                });
                break;
        }
    }//GEN-LAST:event_vistaMouseReleased

    private void vistaMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_vistaMouseClicked
        int x = evt.getX();
        int y = evt.getY();

        LinkedList<Rectangle> areaQeue = networkManager.getAreaQeue();

        float escala = (float) buffImage.getWidth() / (float) vista.getBounds().width;
        switch (areaStatus) {
            case DELETE:
                areaQeue.stream()
                        .filter(a -> a.contains((int) (x * escala), (int) (y * escala)))
                        .findFirst()
                        .ifPresent(a -> areaQeue.removeFirstOccurrence(a));
                areaStatus = SeletionStatus.MODIFY;
                learnArea = areaQeue.getLast();
                break;
            case MODIFY:
                if (leftTopPoint.contains(x, y)) {
                    areaStatus = SeletionStatus.MODIFY_POSITION;
                    break;
                } else if (widthHwightpoint.contains(x, y)) {
                    areaStatus = SeletionStatus.MODIFY_SIZE;
                    break;
                } else {
                    areaQeue.stream()
                            .filter(a -> a.contains((int) (x * escala), (int) (y * escala)))
                            .findFirst()
                            .ifPresent(a -> learnArea = a);
                }
                break;

            case DUPLICATE:
                areaQeue.stream()
                        .filter(a -> a.contains((int) (x * escala), (int) (y * escala)))
                        .findFirst()
                        .ifPresent(a -> {
                            learnArea.width = a.width;
                            learnArea.height = a.height;
                        });
            case ADD:
                areaStatus = SeletionStatus.MODIFY_POSITION;
                break;

        }

        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
        });
    }//GEN-LAST:event_vistaMouseClicked

    private void seleccionCopyActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_seleccionCopyActionPerformed
        seleccion.setSelected(seleccionCopy.isSelected());
        addSelectionButton.setEnabled(seleccion.isSelected());
        duplicaSelectionButton.setEnabled(seleccion.isSelected());
        deleteSelectionButton.setEnabled(seleccion.isSelected());
        
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
        });
    }//GEN-LAST:event_seleccionCopyActionPerformed

    private void seleccionActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_seleccionActionPerformed
        seleccionCopy.setSelected(seleccion.isSelected());
        addSelectionButton.setEnabled(seleccion.isSelected());
        duplicaSelectionButton.setEnabled(seleccion.isSelected());
        deleteSelectionButton.setEnabled(seleccion.isSelected());
        limpiajButton.setEnabled(seleccion.isSelected());
        
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
        });
    }//GEN-LAST:event_seleccionActionPerformed

    private void addSelectionButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_addSelectionButtonActionPerformed
        learnArea = new Rectangle();
        LinkedList<Rectangle> areaQeue = networkManager.getAreaQeue();
        learnArea.setSize(100, 100);
        learnArea.setLocation(10, 10);
        areaStatus = SeletionStatus.ADD;
        areaQeue.add(learnArea);
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
        });
    }//GEN-LAST:event_addSelectionButtonActionPerformed

    private void deleteSelectionButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_deleteSelectionButtonActionPerformed
        areaStatus = SeletionStatus.DELETE;
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
        });
    }//GEN-LAST:event_deleteSelectionButtonActionPerformed

    private void cleanCopyActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cleanCopyActionPerformed
        cleanActionPerformed(evt);
    }//GEN-LAST:event_cleanCopyActionPerformed

    private void cargaOriginalActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cargaOriginalActionPerformed
        destBuffImage = buffImage;
        bufferImageFiltered = destBuffImage;
        java.awt.EventQueue.invokeLater(() -> {
            respuesta.repaint();
        });

    }//GEN-LAST:event_cargaOriginalActionPerformed

    private void jButtonSalvaImagenActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButtonSalvaImagenActionPerformed
        int showSaveDialog = jFileChooserImagenSalva.showSaveDialog(null);
        if (showSaveDialog == javax.swing.JFileChooser.APPROVE_OPTION) {
            log.info("jFileChooserImagenSalva");
            try {
                jFileChooserImagenSalva.getSelectedFile().createNewFile();
                ImageIO.write(bufferImageFiltered, "JPEG", jFileChooserImagenSalva.getSelectedFile());
            } catch (IOException ex) {
                log.info("file error", ex);
                javax.swing.JOptionPane.showMessageDialog(null, "Error al guardar imagen", "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
            }

        }
    }//GEN-LAST:event_jButtonSalvaImagenActionPerformed

    private void cargaPreparadaActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cargaPreparadaActionPerformed
        int showOpenDialog = jFileChooserLoadImagen.showOpenDialog(null);
        if (showOpenDialog == javax.swing.JFileChooser.APPROVE_OPTION) {
            try {
                destBuffImage = ImageIO.read(jFileChooserLoadImagen.getSelectedFile());

                bufferImageFiltered = destBuffImage;
                java.awt.EventQueue.invokeLater(() -> {
                    vista.repaint();
                    respuesta.repaint();
                });
            } catch (IOException ex) {
                log.error("error de archivo ", ex);
                javax.swing.JOptionPane.showMessageDialog(null, "Error al leer imagen", "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
            }
        }
    }//GEN-LAST:event_cargaPreparadaActionPerformed

    private void cargaImagenActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cargaImagenActionPerformed
        int showOpenDialog = jFileChooserLoadImagen.showOpenDialog(null);

        if (showOpenDialog == javax.swing.JFileChooser.APPROVE_OPTION) {
            try {
                buffImage = ImageIO.read(jFileChooserLoadImagen.getSelectedFile());

                destBuffImage = buffImage;
                bufferImageFiltered = destBuffImage;
                java.awt.EventQueue.invokeLater(() -> {
                    vista.repaint();
                    respuesta.repaint();
                });
            } catch (IOException ex) {
                log.error("error de archivo ", ex);
                javax.swing.JOptionPane.showMessageDialog(null, "Error al leer imagen", "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
            }
        }

    }//GEN-LAST:event_cargaImagenActionPerformed

    private void iteracionesStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_iteracionesStateChanged
        jProgressBar1.setMaximum((Integer) iteraciones.getValue());
    }//GEN-LAST:event_iteracionesStateChanged

    private void vistaMouseMoved(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_vistaMouseMoved

        if (buffImage == null) {
            return;
        }
        int x = evt.getX();
        int y = evt.getY();

        LinkedList<Rectangle> areaQeue = networkManager.getAreaQeue();
        float escala = (float) buffImage.getWidth() / (float) vista.getBounds().width;

        switch (areaStatus) {
            case MODIFY_POSITION:
                vista.setCursor(new java.awt.Cursor(java.awt.Cursor.MOVE_CURSOR));
                break;
            case MODIFY_SIZE:
                vista.setCursor(new java.awt.Cursor(java.awt.Cursor.SE_RESIZE_CURSOR));
                break;
            case MODIFY:
                if (leftTopPoint.contains(x, y)) {
                    vista.setCursor(new java.awt.Cursor(java.awt.Cursor.MOVE_CURSOR));
                } else if (widthHwightpoint.contains(x, y)) {
                    vista.setCursor(new java.awt.Cursor(java.awt.Cursor.SE_RESIZE_CURSOR));
                } else if (areaQeue.stream()
                        .anyMatch(a -> a.contains((int) (x * escala), (int) (y * escala)))) {

                    vista.setCursor(new java.awt.Cursor(java.awt.Cursor.CROSSHAIR_CURSOR));
                } else {
                    vista.setCursor(new java.awt.Cursor(java.awt.Cursor.DEFAULT_CURSOR));

                }

                break;
            case ADD:
                vista.setCursor(new java.awt.Cursor(java.awt.Cursor.CROSSHAIR_CURSOR));
                break;
            case DUPLICATE:
            case DELETE:
                vista.setCursor(new java.awt.Cursor(java.awt.Cursor.HAND_CURSOR));
                break;
        }
    }//GEN-LAST:event_vistaMouseMoved

    private void vistaMouseDragged(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_vistaMouseDragged
        int x = evt.getX();
        int y = evt.getY();

        switch (areaStatus) {
            case MODIFY_POSITION:
                vista.setCursor(new java.awt.Cursor(java.awt.Cursor.MOVE_CURSOR));
                break;
            case MODIFY_SIZE:
                vista.setCursor(new java.awt.Cursor(java.awt.Cursor.SE_RESIZE_CURSOR));
                break;
            case MODIFY:
                vista.setCursor(new java.awt.Cursor(java.awt.Cursor.DEFAULT_CURSOR));
                if (leftTopPoint.contains(x, y)) {
                    areaStatus = SeletionStatus.MODIFY_POSITION;
                } else if (widthHwightpoint.contains(x, y)) {
                    areaStatus = SeletionStatus.MODIFY_SIZE;
                }
                break;
        }
    }//GEN-LAST:event_vistaMouseDragged

    private void jCheckBoxScale1neg1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jCheckBoxScale1neg1ActionPerformed
        libreJRadioButton.setEnabled(!jCheckBoxScale1neg1.isSelected());
        preventJRadioButton.setEnabled(!jCheckBoxScale1neg1.isSelected());
        if (libreJRadioButton.isSelected() || preventJRadioButton.isSelected()) {
            scaleJRadioButton.setSelected(true);
        }
    }//GEN-LAST:event_jCheckBoxScale1neg1ActionPerformed

    private void jPanel3ComponentShown(java.awt.event.ComponentEvent evt) {//GEN-FIRST:event_jPanel3ComponentShown
        CardLayout cl = (CardLayout) jPanelCard.getLayout();
        cl.show(jPanelCard, "cardPesos");
    }//GEN-LAST:event_jPanel3ComponentShown

    private void jPanel3ComponentHidden(java.awt.event.ComponentEvent evt) {//GEN-FIRST:event_jPanel3ComponentHidden
        CardLayout cl = (CardLayout) jPanelCard.getLayout();
        cl.first(jPanelCard);
    }//GEN-LAST:event_jPanel3ComponentHidden

    private void jButtonAddRowActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButtonAddRowActionPerformed
        DefaultTableModel model = (DefaultTableModel) jTableWeight.getModel();
        Object[] fila = new Object[]{Integer.valueOf(12), Boolean.FALSE, WeightCreationEnum.RANDOM, WeightModelingEnum.NORMALIZED, ActivationFunctionEnum.LINEAL,  Float.valueOf(5.0E-5F), BasicLearningEstrategyEnum.ONE_ADV_ONE_TREE_BACK_ONE};
        int selectedRow = jTableWeight.getSelectedRow();
        if (selectedRow == -1) {
            model.addRow(fila);

        } else {
            model.insertRow(selectedRow, fila);

        }
    }//GEN-LAST:event_jButtonAddRowActionPerformed

    private void jButtonRemoveRowActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButtonRemoveRowActionPerformed
        DefaultTableModel model = (DefaultTableModel) jTableWeight.getModel();

        if (model.getRowCount() > 1) {
            int selectedRow = jTableWeight.getSelectedRow();
            model.removeRow(selectedRow);
        }
    }//GEN-LAST:event_jButtonRemoveRowActionPerformed

    private void inNeursStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_inNeursStateChanged
        jLabelNumInPixels.setText(getRawPixelInNeur());
    }//GEN-LAST:event_inNeursStateChanged

    private void inNeursComponentShown(java.awt.event.ComponentEvent evt) {//GEN-FIRST:event_inNeursComponentShown
        jLabelNumInPixels.setText(getRawPixelInNeur());
    }//GEN-LAST:event_inNeursComponentShown

    private void jTableWeightPropertyChange(java.beans.PropertyChangeEvent evt) {//GEN-FIRST:event_jTableWeightPropertyChange
        jLabelNumOutPixels.setText(getRawPixelOutNeur());
    }//GEN-LAST:event_jTableWeightPropertyChange

    private void duplicaSelectionButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_duplicaSelectionButtonActionPerformed
        learnArea = new Rectangle();
        LinkedList<Rectangle> areaQeue = networkManager.getAreaQeue();
        learnArea.setSize(100, 100);
        learnArea.setLocation(10, 10);
        areaStatus = SeletionStatus.DUPLICATE;
        areaQeue.add(learnArea);
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
        });
    }//GEN-LAST:event_duplicaSelectionButtonActionPerformed

    private void jButtonSalvaRedActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButtonSalvaRedActionPerformed
        log.info("salvando red...");
        int showSaveDialog = jFileChooserSaveNet.showSaveDialog(null);
        if (showSaveDialog == javax.swing.JFileChooser.APPROVE_OPTION) {
            
            String archivo = jFileChooserSaveNet.getSelectedFile().getPath();
            
            if (!archivo.endsWith(".nn3")) {
                archivo = archivo + ".nn3";
            }
            DefaultTableModel dtm = (DefaultTableModel)jTableWeight.getModel();

            try (final OutputStream fos = Files.newOutputStream(Paths.get(archivo)); final BufferedOutputStream out = new BufferedOutputStream(fos); final GzipCompressorOutputStream gzOut = new GzipCompressorOutputStream(out); final ObjectOutputStream oos = new ObjectOutputStream(gzOut)) {
                
                oos.writeBoolean(jCheckBoxScale1neg1.isSelected());
                String buttonText = Collections.list(adaptInputButtonGroup.getElements()).stream()
                        .filter(e -> adaptInputButtonGroup.isSelected(e.getModel()))
                        .map(e -> e.getText())
                        .findFirst().orElseThrow();
                oos.writeUTF(buttonText);
                int inStep = (int) inNeurs.getValue();
                oos.writeInt(inStep);
                oos.writeObject(dtm.getDataVector());
                oos.writeObject(networkManager.getWeights());
                oos.writeObject(networkManager.getAreaQeue());
                oos.close();
            } catch (FileNotFoundException ex) {
                log.error("error al guardar  red", ex);
            } catch (IOException ex) {
                javax.swing.JOptionPane.showMessageDialog(null, "Modelo de red neuronal corrupto", "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                log.error("error al guardar  red", ex);
            }
            log.info("red guardada.");
        }
    }//GEN-LAST:event_jButtonSalvaRedActionPerformed
    
    @SuppressWarnings("unchecked")
    private void jButtonCargaRedActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButtonCargaRedActionPerformed
        log.info("Cargando red...");
        int showSaveDialog = jFileChooserLoadNet.showOpenDialog(null);
        if (showSaveDialog == javax.swing.JFileChooser.APPROVE_OPTION) {
            String archivo = jFileChooserLoadNet.getSelectedFile().getPath();
            if (archivo.endsWith("nn3")) {
                DefaultTableModel dtm = (DefaultTableModel)jTableWeight.getModel();                

                try (final InputStream fis = Files.newInputStream(Paths.get(archivo)); final BufferedInputStream bis = new BufferedInputStream(fis); final GzipCompressorInputStream gzIn = new GzipCompressorInputStream(bis); final ObjectInputStream ois = new ObjectInputStream(gzIn)) {
                    boolean scale1neg1 = ois.readBoolean();
                    String buttonText = ois.readUTF();
                    int inStep = ois.readInt();
                    Vector<Vector> dtm2 = (Vector<Vector>) ois.readObject();
                    NumericMatriz<Float>[] weights = (NumericMatriz<Float>[])ois.readObject();
                    List<Rectangle> areaQeue = (List<Rectangle>)ois.readObject();
                    ois.close();
                    jCheckBoxScale1neg1.setSelected(scale1neg1);
                    Collections.list(adaptInputButtonGroup.getElements()).stream()
                            .filter(e -> e.getText().equals(buttonText))
                            .findFirst().ifPresent( e -> { 
                                e.setSelected(true);
                            });
                    inNeurs.setValue(inStep);
                    networkManager.setInStep(inStep);
                    networkManager.setWeights(weights);
                    networkManager.getAreaQeue().clear();
                    networkManager.getAreaQeue().addAll(areaQeue);
                        
                    if (Objects.nonNull(weights)) {
                        int[] hiddenStep = Arrays.stream(weights).map(NumericMatriz::getDominio).mapToInt(Dominio::getFila).peek(hid -> log.info("neuronas <{}>", hid)).toArray();
                        networkManager.setHiddenStep(hiddenStep);
                    } else {
                        networkManager.setHiddenStep(null);
                    }
                    
                    dtm.setDataVector(dtm2, new Vector<String>(Arrays.asList("Neuronas", "Tendencia", "Creacion Pesos", "Estilo Pesos", "Func. Activacion", "Fact. Aprendisaje", "estratg. Aprendisaje")));
                    
                    if (jTableWeight.getColumnModel().getColumnCount() > 0) {
                        jTableWeight.getColumnModel().getColumn(0).setResizable(false);
                        jTableWeight.getColumnModel().getColumn(0).setCellEditor(getNeuronCellEditor());
                        jTableWeight.getColumnModel().getColumn(1).setResizable(false);
                        jTableWeight.getColumnModel().getColumn(2).setCellEditor(getNeuronCreationWeigth());
                        jTableWeight.getColumnModel().getColumn(3).setResizable(false);
                        jTableWeight.getColumnModel().getColumn(3).setCellEditor(getNeuronStyleWeigth());
                        jTableWeight.getColumnModel().getColumn(4).setResizable(false);
                        jTableWeight.getColumnModel().getColumn(4).setCellEditor(getActivationFunctionCellEditor());
                        jTableWeight.getColumnModel().getColumn(5).setResizable(false);
                        jTableWeight.getColumnModel().getColumn(5).setCellEditor(getLearningFactorCellEditor());
                        jTableWeight.getColumnModel().getColumn(5).setCellRenderer(getLearningFactorCellRender());
                        jTableWeight.getColumnModel().getColumn(6).setResizable(false);
                        jTableWeight.getColumnModel().getColumn(6).setCellEditor(getLearningEstrategyCellEditor());
                    }

                } catch (FileNotFoundException ex) {
                    log.error("error al cargar red", ex);
                    javax.swing.JOptionPane.showMessageDialog(null, "Archivo de red neuronal no encontrado", "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                } catch (IOException | ClassNotFoundException ex) {
                    log.error("error al cargar red", ex);
                    javax.swing.JOptionPane.showMessageDialog(null, "Archivo de red neuronal corrupto", "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                }
            } else {
                javax.swing.JOptionPane.showMessageDialog(null, "Archivo no corresponde a una red neuronal", "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
            }

        }
    }//GEN-LAST:event_jButtonCargaRedActionPerformed

    private void limpiajButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_limpiajButtonActionPerformed
       
        LinkedList<Rectangle> areaQeue = networkManager.getAreaQeue();
        areaQeue.clear();
        learnArea = new Rectangle();
        learnArea.setSize(100, 100);
        learnArea.setLocation(10, 10);
        areaQeue.add(learnArea);
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
        });
    }//GEN-LAST:event_limpiajButtonActionPerformed

    private float calculaMatriz(int i, int j, int kwidth) {
        float retorno;
        float half = (float) kwidth / 2;
        float di = (float) i - half;
        float dj = (float) j - half;

        retorno = 0.5f - (float) 1 / (1 + di * di + dj * dj);
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

        image = new BufferedImage(destCM, wr,
                destCM.isAlphaPremultiplied(), null);

        return image;
    }

    private javax.swing.JPanel getNuevaVista() {
        return new JPanel(true) {

            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                if (Objects.nonNull(buffImage)) {
                    Graphics2D localg = (Graphics2D) g;
                    float escala = (float) vista.getBounds().width / (float) buffImage.getWidth();

                    AffineTransform xforM = AffineTransform.getScaleInstance(escala, escala);
                    AffineTransformOp rop = new AffineTransformOp(xforM, AffineTransformOp.TYPE_BILINEAR);
                    localg.drawImage(buffImage, rop, 0, 0);
                    LinkedList<Rectangle> areaQeue = networkManager.getAreaQeue();

                    if (seleccion.isSelected()){
                        areaQeue.forEach(a -> {
                            Rectangle evalArea = new Rectangle(a);
                            evalArea.x = (int) (evalArea.x * escala);
                            evalArea.y = (int) (evalArea.y * escala);
                            evalArea.width = (int) (evalArea.width * escala);
                            evalArea.height = (int) (evalArea.height * escala);
                            switch (areaStatus) {
                                case MODIFY:
                                    if (a.equals(learnArea)) {
                                        leftTopPoint.setSize(10, 10);
                                        leftTopPoint.setLocation(evalArea.x - 5, evalArea.y - 5);

                                        widthHwightpoint.setSize(10, 10);
                                        widthHwightpoint.setLocation(
                                                evalArea.width + evalArea.x - 5,
                                                evalArea.height + evalArea.y - 5);

                                        localg.setColor(Color.RED);
                                    } else {
                                        localg.setColor(Color.BLACK);
                                    }
                                    break;
                                case MODIFY_POSITION:
                                case MODIFY_SIZE:
                                    if (a.equals(learnArea)) {
                                        localg.setColor(Color.BLUE);
                                    } else {
                                        localg.setColor(Color.BLACK);
                                    }
                                    break;
                                case DELETE:
                                    localg.setColor(Color.RED);
                                    break;
                                case ADD:
                                    localg.setColor(Color.BLACK);
                                    break;
                                case DUPLICATE:
                                    localg.setColor(Color.YELLOW);
                                    break;

                            }
                            localg.draw(evalArea);
                            if (SeletionStatus.MODIFY.equals(areaStatus)) {
                                localg.setColor(Color.WHITE);
                                localg.draw(leftTopPoint);
                                localg.setColor(Color.GREEN);
                                localg.draw(widthHwightpoint);
                            }

                        });
                        
                    }

                }
            }

        };
    }

    private javax.swing.JPanel getNuevaRespuesta() {
        return new JPanel(true) {

            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                if (Objects.nonNull(getBufferImageFiltered())) {
                    Graphics2D localg = (Graphics2D) g;
                    float escala = (float) respuesta.getBounds().width / (float) getBufferImageFiltered().getWidth();
                    AffineTransform xforM = AffineTransform.getScaleInstance(escala, escala);
                    AffineTransformOp rop = new AffineTransformOp(xforM, AffineTransformOp.TYPE_BILINEAR);
                    localg.drawImage(getBufferImageFiltered(), rop, 0, 0);

                }
            }

        };
    }

    private javax.swing.JPanel getNuevaErrorGram() {
        return new JPanel(true) {
            @Override
            protected void paintComponent(Graphics grphcs) {
                super.paintComponent(grphcs);
                NumericMatriz<Float> errorGraph = networkManager.getErrorGraph();

                if (Objects.nonNull(errorGraph)) {
                    synchronized (errorGraph) {
                        OptionalDouble maxError = errorGraph.values().stream().mapToDouble((i) -> i.doubleValue()).average();
                        List<ParOrdenado> proccesDomain = networkManager.getProccesDomain();
                        double size = (double) proccesDomain.size();

                        Graphics2D gr2 = (Graphics2D) grphcs;
                        gr2.setColor(Color.RED);
                        double tol = maxError.orElse(0.0001);
                        double lcWidth = ((double)jErrorGraf.getWidth()) / size;
                        double lclHeight = ((double)jErrorGraf.getHeight()) / tol;
                        gr2.translate(0, 0);
                        gr2.scale(lcWidth, lclHeight);
                        int adv = 0;

                        for (ParOrdenado idx : proccesDomain) {
                            Double errorPoint = (Double) errorGraph.get(idx).doubleValue();
                            Shape shape = new Rectangle2D.Double(adv++, 0, 1, errorPoint);
                            gr2.draw(shape);

                        }
                    }
                }
            }

        };
    }

    public FileNameExtensionFilter getFileNameNetworkExtensionFilter() {
        return fileNameNetworkExtensionFilter;
    }

    private void enableConsole() {
        procesar.setEnabled(true);
        cleanCopy.setEnabled(true);
        clean.setEnabled(true);
        entrenar.setEnabled(true);
        jCheckBoxParalell.setEnabled(true);
        freno.setSelected(false);
        jComboBoxBufferSize.setEnabled(true);
        jTableWeight.setEnabled(true);
        jButtonCargaRed.setEnabled(true);
        jButtonSalvaRed.setEnabled(true);
        jButtonAddRow.setEnabled(true);
        jButtonRemoveRow.setEnabled(true);
        inNeurs.setEnabled(true);
        normaExtJRadioButton.setEnabled(true);
        cargaOriginal.setEnabled(true);
        cargaPreparada.setEnabled(true);
        suavizaResultado.setEnabled(true);
        enmascaraResultado.setEnabled(true);
    }

    private void disableConsole() {
       
        procesar.setEnabled(false);
        cleanCopy.setEnabled(false);
        clean.setEnabled(false);
        entrenar.setEnabled(false);
        jCheckBoxParalell.setEnabled(false);
        jComboBoxBufferSize.setEnabled(false);
        jTableWeight.setEnabled(false);
        jButtonCargaRed.setEnabled(false);
        jButtonSalvaRed.setEnabled(false);
        jButtonAddRow.setEnabled(false);
        jButtonRemoveRow.setEnabled(false);
        inNeurs.setEnabled(false);
        normaExtJRadioButton.setEnabled(false);
        cargaOriginal.setEnabled(false);
        cargaPreparada.setEnabled(false);
        suavizaResultado.setEnabled(false);
        enmascaraResultado.setEnabled(false);
    }

    private enum SeletionStatus {
        MODIFY, MODIFY_POSITION, MODIFY_SIZE, ADD, DELETE, DUPLICATE
    }

    public static ComboBoxModel getActivationComboBoxModel() {
        return NeuronStyleTableCellEditorImpl.getModelingComboBoxModel();
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
    private javax.swing.ButtonGroup adaptInputButtonGroup;
    private javax.swing.JButton addSelectionButton;
    private javax.swing.JButton cargaImagen;
    private javax.swing.JButton cargaOriginal;
    private javax.swing.JButton cargaPreparada;
    private javax.swing.JButton clean;
    private javax.swing.JButton cleanCopy;
    private javax.swing.JButton deleteSelectionButton;
    private javax.swing.JButton duplicaSelectionButton;
    private javax.swing.JButton enmascaraResultado;
    private javax.swing.JCheckBox entrenar;
    private javax.swing.JToggleButton freno;
    private javax.swing.JSpinner inNeurs;
    private javax.swing.JSpinner iteraciones;
    private javax.swing.JButton jButtonAddRow;
    private javax.swing.JButton jButtonCargaRed;
    private javax.swing.JButton jButtonRemoveRow;
    private javax.swing.JButton jButtonSalvaImagen;
    private javax.swing.JButton jButtonSalvaRed;
    private javax.swing.JCheckBox jCheckBoxParalell;
    private javax.swing.JCheckBox jCheckBoxScale1neg1;
    private javax.swing.JComboBox<String> jComboBoxBufferSize;
    private javax.swing.JPanel jErrorGraf;
    private javax.swing.JFileChooser jFileChooserImagenSalva;
    private javax.swing.JFileChooser jFileChooserLoadImagen;
    private javax.swing.JFileChooser jFileChooserLoadImagenResult;
    private javax.swing.JFileChooser jFileChooserLoadNet;
    private javax.swing.JFileChooser jFileChooserSaveNet;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabelNumInPixels;
    private javax.swing.JLabel jLabelNumOutPixels;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JPanel jPanel3;
    private javax.swing.JPanel jPanel4;
    private javax.swing.JPanel jPanel5;
    private javax.swing.JPanel jPanelCard;
    private javax.swing.JPanel jPanelTop;
    private javax.swing.JProgressBar jProgressBar1;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JSeparator jSeparator1;
    private javax.swing.JSeparator jSeparator2;
    private javax.swing.JSplitPane jSplitPane1;
    private javax.swing.JTabbedPane jTabbedPane1;
    private javax.swing.JTable jTableWeight;
    private javax.swing.JRadioButton libreJRadioButton;
    private javax.swing.JButton limpiajButton;
    private javax.swing.JRadioButton normaExtJRadioButton;
    private javax.swing.JRadioButton normalizeJRadioButton;
    private javax.swing.JRadioButton preventJRadioButton;
    private javax.swing.JButton procesar;
    private javax.swing.JRadioButton reflectJRadioButton;
    private javax.swing.JPanel respuesta;
    private javax.swing.JRadioButton scaleJRadioButton;
    private javax.swing.JCheckBox seleccion;
    private javax.swing.JCheckBox seleccionCopy;
    private javax.swing.JButton suavizaResultado;
    private javax.swing.JPanel vista;
    // End of variables declaration//GEN-END:variables

    public FileNameExtensionFilter getFileNameExtensionFilter() {
        return fileNameExtensionFilter;
    }

    public void addFileNameExtensionFilter(javax.swing.JFileChooser jFileChooser) {
        jFileChooser.addChoosableFileFilter(fileNameExtensionFilter);
    }

    public FileNameExtensionFilter getFileNameExtensionFilterImage() {
        return fileNameExtensionFilterImage;
    }

    public BufferedImage getBufferImageFiltered() {
        return bufferImageFiltered;
    }

    private static class LearningFactorTableCellEditorImpl extends AbstractCellEditor implements TableCellEditor {

        private JSpinner factor = new JSpinner(getSpinnerModel());

        public LearningFactorTableCellEditorImpl() {
        }

        @Override
        public Object getCellEditorValue() {
            return (Float) factor.getValue();
        }

        @Override
        public Component getTableCellEditorComponent(JTable jtable, Object value, boolean isSelected, int rowIndex, int colIndex) {
            if (Objects.nonNull(value)) {
                factor.setValue((Float) value);
            }
            return factor;
        }

        public static SpinnerModel getSpinnerModel() {
            SpinnerListModel spinnerModel = new SpinnerListModel(LearningEstrategy.floatBasicLearningSeries);
            int valorMedio = LearningEstrategy.floatBasicLearningSeries.length / 2;
            spinnerModel.setValue(LearningEstrategy.floatBasicLearningSeries[valorMedio]);
            return spinnerModel;
        }

    }

    private static class NeuronTableCellEditorImpl extends AbstractCellEditor implements TableCellEditor {

        private JSpinner neuronas = new JSpinner(new SpinnerNumberModel(15, 1, 2049, 3));
        private JSpinner alCuadrado = new JSpinner(getCuadradoSpinnerModel());
        private boolean isCuadrado = false;

        public static SpinnerModel getCuadradoSpinnerModel() {
            return new SpinnerListModel(IntStream.range(1, 500).map(i -> i * i * 3).boxed().toArray(Integer[]::new));
        }

        public NeuronTableCellEditorImpl() {
        }

        @Override
        public Object getCellEditorValue() {
            if (isCuadrado) {
                return alCuadrado.getValue();
            }
            return neuronas.getValue();
        }

        @Override
        public Component getTableCellEditorComponent(JTable jtable, Object value, boolean isSelected, int rowIndex, int colIndex) {
            isCuadrado = jtable.getRowCount() == rowIndex + 1;
            if (isCuadrado) {
                alCuadrado.setValue((Integer) value);
                return alCuadrado;
            }
            neuronas.setValue((Integer) value);
            return neuronas;
        }

    }

    private static class NeuronStyleTableCellEditorImpl extends AbstractCellEditor implements TableCellEditor {

        public static ComboBoxModel getModelingComboBoxModel() {
            return new DefaultComboBoxModel<>(WeightModelingEnum.values());
        }

        private final javax.swing.JComboBox<WeightModelingEnum> style;

    @SuppressWarnings("unchecked")
        public NeuronStyleTableCellEditorImpl() {
            this.style = new javax.swing.JComboBox<>(getModelingComboBoxModel());
            style.setSelectedIndex(0);
        }

        @Override
        public Object getCellEditorValue() {
            return style.getSelectedItem();
        }

        @Override
        public Component getTableCellEditorComponent(JTable jtable, Object o, boolean bln, int i, int i1) {
            if (Objects.isNull(o)) {
                style.setSelectedIndex(0);
            } else {
                style.setSelectedItem(o);
            }
            return style;
        }
    }

    private static class CreationWeightTableCellEditorImpl extends AbstractCellEditor implements TableCellEditor {

        public static ComboBoxModel getCreationComboBoxModel() {
            return new DefaultComboBoxModel<>(WeightCreationEnum.values());
        }

    @SuppressWarnings("unchecked")
        public CreationWeightTableCellEditorImpl() {
            creation = new JComboBox<>(getCreationComboBoxModel());
            creation.setSelectedIndex(0);
        }

        private final javax.swing.JComboBox<WeightCreationEnum> creation;

        @Override
        public Object getCellEditorValue() {
            return creation.getSelectedItem();
        }

        @Override
        public Component getTableCellEditorComponent(JTable jtable, Object o, boolean bln, int i, int i1) {
            if (Objects.isNull(o)) {
                creation.setSelectedIndex(0);
            } else {
                creation.setSelectedItem(o);
            }
            return creation;
        }
    }

    private static class ActivationFunctionTableCellEditorImpl extends AbstractCellEditor implements TableCellEditor {

        public static ComboBoxModel getActivationComboBoxModel() {
            return new DefaultComboBoxModel<>(ActivationFunctionEnum.values());
        }

        @SuppressWarnings("unchecked")
        public ActivationFunctionTableCellEditorImpl() {
            activation = new JComboBox<>(getActivationComboBoxModel());
            activation.setSelectedIndex(0);
        }

        private final javax.swing.JComboBox<ActivationFunctionEnum> activation;

        @Override
        public Object getCellEditorValue() {
            return activation.getSelectedItem();
        }

        @Override
        public Component getTableCellEditorComponent(JTable jtable, Object o, boolean bln, int i, int i1) {
            if (Objects.isNull(o)) {
                activation.setSelectedIndex(0);
            } else {
                activation.setSelectedItem(o);
            }
            return activation;
        }
    }

    private static class LearningEstrategyTableCellEditorImpl extends AbstractCellEditor implements TableCellEditor {

        public static ComboBoxModel getLearninEstrategyComboBoxModel() {
            return new DefaultComboBoxModel<>(BasicLearningEstrategyEnum.values());
        }

        private final javax.swing.JComboBox<ActivationFunctionEnum> strategy;

        @SuppressWarnings("unchecked")
        public LearningEstrategyTableCellEditorImpl() {
            this.strategy = new JComboBox<>(getLearninEstrategyComboBoxModel());
            this.strategy.setSelectedIndex(0);
        }

        @Override
        public Object getCellEditorValue() {
            return strategy.getSelectedItem();
        }

        @Override
        public Component getTableCellEditorComponent(JTable jtable, Object o, boolean bln, int i, int i1) {
            if (Objects.isNull(o)) {
                strategy.setSelectedIndex(0);
                return strategy;
            }
            strategy.setSelectedItem(o);
            return strategy;
        }

    }

    private static class LearningFactorCellRenderImp extends DefaultTableCellRenderer implements TableCellRenderer, Serializable {

        @Override
        public Component getTableCellRendererComponent(JTable jtable, Object o, boolean bln, boolean bln1, int i, int i1) {
            return super.getTableCellRendererComponent(jtable, String.format("%1.1E", (Float) o), bln, bln1, i, i1);
        }

    }
}
