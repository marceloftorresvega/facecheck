/*
 * The MIT License
 *
 * Copyright 2019 lorenzo.
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
package org.tensa.facecheck.filter;

import java.awt.RenderingHints;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.ColorConvertOp;
import java.awt.image.ColorModel;
import java.awt.image.IndexColorModel;
import java.awt.image.WritableRaster;
import java.util.stream.IntStream;

/**
 *
 * @author lorenzo
 */
public class MaskOp implements BufferedImageOp{

    private BufferedImage otherSrc;
    
    @Override
    public BufferedImage filter(BufferedImage src, BufferedImage dest) {
        if (src == null) {
            throw new NullPointerException("src image is null");
        }
        if (otherSrc == null) {
            throw new NullPointerException("other src image is null");
        }
        if (src == dest) {
            throw new IllegalArgumentException("src image cannot be the "+
                                               "same as the dest image");
        }
        
        boolean needToConvert = false;
        ColorModel srcCM = src.getColorModel();
        ColorModel destCM;
        BufferedImage origDst = dest;

        // Can't convolve an IndexColorModel.  Need to expand it
        if (srcCM instanceof IndexColorModel) {
            IndexColorModel icm = (IndexColorModel) srcCM;
            src = icm.convertToIntDiscrete(src.getRaster(), false);
            srcCM = src.getColorModel();
        }

        if (dest == null) {
            dest = createCompatibleDestImage(src, null);
            destCM = srcCM;
            origDst = dest;
        }
        else {
            destCM = dest.getColorModel();
            if (srcCM.getColorSpace().getType() !=
                destCM.getColorSpace().getType())
            {
                needToConvert = true;
                dest = createCompatibleDestImage(src, null);
                destCM = dest.getColorModel();
            }
            else if (destCM instanceof IndexColorModel) {
                dest = createCompatibleDestImage(src, null);
                destCM = dest.getColorModel();
            }
        }
        
        int width = src.getWidth();
        int height = src.getHeight();
        int step = 50;
        
        WritableRaster srcRaster = src.getRaster();
        WritableRaster otherRaster = otherSrc.getRaster();
        
        WritableRaster destRaster = dest.getRaster();
        
        IntStream.iterate(0, s -> s + step)
                .limit(width / step)
                .filter(i -> i < width).parallel()
                .forEach(i -> {
                    IntStream.iterate(0, s -> s + step)
                            .limit(height / step)
                            .filter( j -> j < height)
                            .forEach( j -> {
                                double[] srcPixels = srcRaster.getPixels(i, j, step, step, (double[]) null);
                                double[] otherPixels = otherRaster.getPixels(i, j, step, step, (double[]) null);
                                double[] destPixels = new double[3 * step * step];
                                
                                for(int k = 0; k< 3 * step * step; k++){
                                    if(otherPixels[k] == 0.0) {
                                        destPixels[k] = srcPixels[k];
                                    } else {
                                        destPixels[k] = otherPixels[k];
                                        
                                    }
                                }
                                
                                destRaster.setPixels(i, j, step, step, destPixels);
                            
                            });
                            
                });
        
        if (needToConvert) {
            ColorConvertOp ccop = new ColorConvertOp(null);
            ccop.filter(dest, origDst);
        }
        else if (origDst != dest) {
            java.awt.Graphics2D g = origDst.createGraphics();
            try {
                g.drawImage(dest, 0, 0, null);
            } finally {
                g.dispose();
            }
        }

        return origDst;
    }

    @Override
    public Rectangle2D getBounds2D(BufferedImage src) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
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

    @Override
    public Point2D getPoint2D(Point2D srcPt, Point2D destPt) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public RenderingHints getRenderingHints() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    /**
     * @return the otherSrc
     */
    public BufferedImage getOtherSrc() {
        return otherSrc;
    }

    /**
     * @param otherSrc the otherSrc to set
     */
    public void setOtherSrc(BufferedImage otherSrc) {
        this.otherSrc = otherSrc;
    }
    
}
