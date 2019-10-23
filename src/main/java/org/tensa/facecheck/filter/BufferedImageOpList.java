package org.tensa.facecheck.filter;

import java.awt.RenderingHints;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.ColorModel;
import java.util.LinkedList;

/**
 *
 * @author lorenzo
 */
public class BufferedImageOpList extends LinkedList<BufferedImageOp> implements BufferedImageOp {

    @Override
    public BufferedImage filter(BufferedImage src, BufferedImage dest) {
        BufferedImage actual = src;
        for ( BufferedImageOp op : this ) {
            if (op.equals(this.getLast())) {
                break;
            }
            actual = op.filter(actual, null);
        }
        actual = this.getLast().filter(actual, dest);
        return actual;
    }

    @Override
    public Rectangle2D getBounds2D(BufferedImage src) {
        return this.getLast().getBounds2D(src);
    }

    @Override
    public BufferedImage createCompatibleDestImage(BufferedImage src, ColorModel destCM) {
        return this.getLast().createCompatibleDestImage(src, destCM);
    }

    @Override
    public Point2D getPoint2D(Point2D srcPt, Point2D dstPt) {
        return this.getLast().getPoint2D(srcPt, dstPt);
    }

    @Override
    public RenderingHints getRenderingHints() {
        return this.getLast().getRenderingHints();
    }
    
}
