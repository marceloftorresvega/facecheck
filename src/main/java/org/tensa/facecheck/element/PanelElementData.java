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
package org.tensa.facecheck.element;

/**
 * Gestiona ventanas o paneles de elementos ElementData para mantener el orden o
 * ubicacion fisica de estos incluso en interacciones recursivas
 *
 * @author Marcelo
 */
public interface PanelElementData extends ElementData {

    /**
     * genera un panel delimitado por los parametros de posicion, ancho y alto,
     * los que no deben exceder los del panel contenedor
     *
     * @param x posicion superior isquierda del panel
     * @param y posicion superior isquierda del panel
     * @param w ancho del panel
     * @param h alto del panel
     * @return PanelElementData
     */
    PanelElementData getSubPanel(int x, int y, int w, int h);

    /**
     * copia la data en la ubicacion del panel delimitado por los parametros de
     * posicion, ancho y alto, los que no deben exceder los del panel contenedor
     *
     * @param x posicion superior isquierda del panel
     * @param y posicion superior isquierda del panel
     * @param w ancho del panel
     * @param h alto del panel
     * @param panel PanelElementData
     */
    void setSubPanel(int x, int y, int w, int h, PanelElementData panel);

    /**
     * copia la data en la ubicacion del panel delimitado por los parametros de
     * posicion, con el ancho y alto determinados por la data, los que no deben
     * exceder los del panel contenedor
     *
     * @param x posicion superior isquierda del panel
     * @param y posicion superior isquierda del panel
     * @param panel PanelElementData
     */
    void setSubPanel(int x, int y, PanelElementData panel);

    /**
     * genera un panel delimitado por los parametros de posicion, con ancho y
     * alto 1 (un unico elementData)
     *
     * @param x posicion superior isquierda del panel
     * @param y posicion superior isquierda del panel
     * @return PanelElementData
     */
    PanelElementData getElement(int x, int y);

    /**
     * copia la data en la ubicacion del panel delimitado por los parametros de
     * posicion, con el ancho y alto 1 (un unico elementData)
     *
     * @param x posicion superior isquierda del panel
     * @param y posicion superior isquierda del panel
     * @param panel PanelElementData
     */
    void setElement(int x, int y, PanelElementData panel);

}
