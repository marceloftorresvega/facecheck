package org.tensa.facecheck.main;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ConvolveOp;
import java.awt.image.IndexColorModel;
import java.awt.image.Kernel;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Objects;
import javax.imageio.ImageIO;
import javax.swing.ComboBoxModel;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JPanel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensa.facecheck.filter.MaskOp;
import org.tensa.facecheck.layer.impl.HiddenLayer;
import org.tensa.facecheck.layer.impl.PixelByteExpandedLeanringLayer;
import org.tensa.facecheck.layer.impl.PixelsByteExpandedOutputLayer;
import org.tensa.facecheck.layer.impl.SimplePixelsByteExpandedInputLayer;
import org.tensa.tensada.matrix.Dominio;
import org.tensa.tensada.matrix.DoubleMatriz;

/**
 *
 * @author lorenzo
 */
public class VisLoad extends javax.swing.JFrame {

    private final Logger log = LoggerFactory.getLogger(VisLoad.class);

    private final String baseUrl = "\\img\\originales\\";
    private final String testBaseUrl = "\\img\\procesadas\\";
    
    private final String[] imageName = {"IMG_2869", "IMG_2918","IMG_3071","IMG_3076","IMG_3078","IMG_3079"};

    private final String sufxType = ".jpg";
    private ComboBoxModel comboModel;
    private BufferedImage buffImage ;
    private BufferedImage destBuffImage ;
    private final int kwidth = 27;
    private float[] data;
    private BufferedImage bufferImageFiltered;

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
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jComboBox1 = new javax.swing.JComboBox();
        jButton3 = new javax.swing.JButton();
        entrenar = new javax.swing.JButton();
        jSplitPane1 = new javax.swing.JSplitPane();
        vista = getNuevaVista();
        respuesta = getNuevaRespuesta();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Vista de carga");

        jButton1.setText("Carga...");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jButton2.setText("Modifica");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        jComboBox1.setModel(getComboModel());

        jButton3.setText("enmascara");
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });

        entrenar.setText("entrena");
        entrenar.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                entrenarActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jButton1)
                .addGap(18, 18, 18)
                .addComponent(jButton2)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jButton3)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(entrenar)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(jComboBox1, javax.swing.GroupLayout.PREFERRED_SIZE, 131, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(90, 90, 90))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jButton1)
                    .addComponent(jButton2)
                    .addComponent(jComboBox1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jButton3)
                    .addComponent(entrenar))
                .addGap(0, 6, Short.MAX_VALUE))
        );

        vista.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        javax.swing.GroupLayout vistaLayout = new javax.swing.GroupLayout(vista);
        vista.setLayout(vistaLayout);
        vistaLayout.setHorizontalGroup(
            vistaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 316, Short.MAX_VALUE)
        );
        vistaLayout.setVerticalGroup(
            vistaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 240, Short.MAX_VALUE)
        );

        jSplitPane1.setLeftComponent(vista);

        respuesta.setBorder(javax.swing.BorderFactory.createEmptyBorder(1, 1, 1, 1));

        javax.swing.GroupLayout respuestaLayout = new javax.swing.GroupLayout(respuesta);
        respuesta.setLayout(respuestaLayout);
        respuestaLayout.setHorizontalGroup(
            respuestaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 281, Short.MAX_VALUE)
        );
        respuestaLayout.setVerticalGroup(
            respuestaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 240, Short.MAX_VALUE)
        );

        jSplitPane1.setRightComponent(respuesta);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(layout.createSequentialGroup()
                .addGap(15, 15, 15)
                .addComponent(jSplitPane1)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jSplitPane1)
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        String _filename1 = System.getProperty("user.dir") + baseUrl + (String)jComboBox1.getSelectedItem() + sufxType;
        String _filename2 = System.getProperty("user.dir") + testBaseUrl + (String)jComboBox1.getSelectedItem() + sufxType;
        log.info("directorio user <{}>",System.getProperty("user.dir"));
        
        try {
            buffImage = ImageIO.read(new File(_filename1));
            
            destBuffImage = ImageIO.read(new File(_filename2));
            
            java.awt.EventQueue.invokeLater(() -> {
                vista.repaint();
            });
        } catch (IOException ex) {
            log.error("error de archivo <{}>", _filename1, ex);
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        
        
        ConvolveOp conv = new ConvolveOp(new Kernel(kwidth, kwidth, data));
        if(Objects.nonNull(bufferImageFiltered))
            bufferImageFiltered.flush();
        
        bufferImageFiltered = conv.filter(buffImage, null);
        
        java.awt.EventQueue.invokeLater(() -> {
            respuesta.repaint();
        });
    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        
        MaskOp conv = new MaskOp();
        conv.setOtherSrc(destBuffImage);
        if(Objects.nonNull(bufferImageFiltered))
            bufferImageFiltered.flush();
        
        bufferImageFiltered = conv.filter(buffImage, null);
        
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
            respuesta.repaint();
        });
    }//GEN-LAST:event_jButton3ActionPerformed

    private void entrenarActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_entrenarActionPerformed
            log.info("iniciando 0...");
        int step = 501;
        DoubleMatriz weightsH = (DoubleMatriz)new DoubleMatriz(new Dominio(256, 51*51*3)).matrizUno();
            log.info("iniciando 1...");
        DoubleMatriz weightsO = (DoubleMatriz)new DoubleMatriz(new Dominio(256, 501*501*3));
            log.info("iniciando 2...");
        bufferImageFiltered = createCompatibleDestImage(buffImage, null);
            log.info("iniciando 3...");
        
        SimplePixelsByteExpandedInputLayer simplePixelsInputLayer = new SimplePixelsByteExpandedInputLayer();
        SimplePixelsByteExpandedInputLayer simplePixelsCompareLayer = new SimplePixelsByteExpandedInputLayer();
        PixelByteExpandedLeanringLayer pixelLeanringLayer = new PixelByteExpandedLeanringLayer(weightsO, 0.01);
        HiddenLayer hiddenLayer = new HiddenLayer(weightsH, 0.001);
        PixelsByteExpandedOutputLayer pixelsOutputLayer = new PixelsByteExpandedOutputLayer(weightsO);
        
        
        simplePixelsInputLayer.getConsumers().add(hiddenLayer);
        hiddenLayer.getConsumers().add(pixelLeanringLayer);
        hiddenLayer.getConsumers().add(pixelsOutputLayer);
        
        int width = buffImage.getWidth();
        int height = buffImage.getHeight();
        
            log.info("procesando...");
        for(int i=0;i<width;i+=step) {
            log.info("bloque <{}>", i);
            
//            for(int j=0;j<height;j+=step) {
            for(int j=0;j<step+1;j+=step) {
            log.info("sub bloque <{}>", j);
                
            log.info("cargando bloque comparacion <{}><{}>", i, j);
                BufferedImage comp = destBuffImage.getSubimage(i, j, step, step);
                simplePixelsCompareLayer.setSrc(comp);
                simplePixelsCompareLayer.startProduction();
                pixelLeanringLayer.setCompareToLayer(simplePixelsCompareLayer.getOutputLayer());
                
                pixelsOutputLayer.setDest(bufferImageFiltered.getSubimage(i, j, step, step));
                
            log.info("cargando bloque ejecucion <{}><{}>", i, j);
                BufferedImage src = buffImage.getSubimage(i, j, step, step);
                simplePixelsInputLayer.setSrc(src);
                simplePixelsInputLayer.startProduction();
                
            }
            java.awt.EventQueue.invokeLater(() -> {
                vista.repaint();
                respuesta.repaint();
            });
        }
    }//GEN-LAST:event_entrenarActionPerformed

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
    private javax.swing.JButton entrenar;
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JComboBox jComboBox1;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JSplitPane jSplitPane1;
    private javax.swing.JPanel respuesta;
    private javax.swing.JPanel vista;
    // End of variables declaration//GEN-END:variables
}
