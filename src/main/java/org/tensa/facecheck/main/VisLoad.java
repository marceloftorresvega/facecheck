package org.tensa.facecheck.main;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;
import java.io.File;
import java.io.IOException;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.ComboBoxModel;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JPanel;
import org.tensa.facecheck.filter.MaskOp;

/**
 *
 * @author lorenzo
 */
public class VisLoad extends javax.swing.JFrame {

    private final String baseUrl = "C:\\Users\\lorenzo\\Pictures\\camp de invierno\\";
    private final String testBaseUrl = "C:\\Users\\lorenzo\\Pictures\\procesadas\\";
//"IMG_2853", "IMG_2854", "IMG_2855", 
    private final String[] imageName = {"IMG_2869", "IMG_2918","IMG_3071","IMG_3076","IMG_3078","IMG_3079"};

    private final String sufxType = ".jpg";

    private ComboBoxModel comboModel;
    private BufferedImage buffImage ;
    private BufferedImage destBuffImage ;
    private final int kwidth = 27;
    private float[] data;

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
        vista = getNuevaVista();

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

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jButton1)
                .addGap(18, 18, 18)
                .addComponent(jButton2)
                .addGap(18, 18, 18)
                .addComponent(jButton3)
                .addGap(30, 30, 30)
                .addComponent(jComboBox1, javax.swing.GroupLayout.PREFERRED_SIZE, 131, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(185, Short.MAX_VALUE))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jButton1)
                    .addComponent(jButton2)
                    .addComponent(jComboBox1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jButton3))
                .addGap(0, 6, Short.MAX_VALUE))
        );

        vista.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        javax.swing.GroupLayout vistaLayout = new javax.swing.GroupLayout(vista);
        vista.setLayout(vistaLayout);
        vistaLayout.setHorizontalGroup(
            vistaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );
        vistaLayout.setVerticalGroup(
            vistaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 209, Short.MAX_VALUE)
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(layout.createSequentialGroup()
                .addGap(14, 14, 14)
                .addComponent(vista, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(vista, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addGap(33, 33, 33))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        String _filename = baseUrl + (String)jComboBox1.getSelectedItem() + sufxType;
        String _filename2 = testBaseUrl + (String)jComboBox1.getSelectedItem() + sufxType;
        try {
            buffImage = ImageIO.read(new File(_filename));
            
            destBuffImage = ImageIO.read(new File(_filename2));
            
            java.awt.EventQueue.invokeLater(() -> {
                vista.repaint();
            });
        } catch (IOException ex) {
            Logger.getLogger(VisLoad.class.getName()).log(Level.SEVERE, null, ex);
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        
        
        ConvolveOp conv = new ConvolveOp(new Kernel(kwidth, kwidth, data));
        BufferedImage bufferImageFiltered = conv.filter(buffImage, null);
        buffImage.flush();
        buffImage = bufferImageFiltered;
        
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
        });
    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        
        MaskOp conv = new MaskOp();
        conv.setOtherSrc(destBuffImage);
        
        BufferedImage bufferImageFiltered = conv.filter(buffImage, null);
        buffImage.flush();
        buffImage = bufferImageFiltered;
        
        java.awt.EventQueue.invokeLater(() -> {
            vista.repaint();
        });
    }//GEN-LAST:event_jButton3ActionPerformed

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
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JComboBox jComboBox1;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel vista;
    // End of variables declaration//GEN-END:variables
}
