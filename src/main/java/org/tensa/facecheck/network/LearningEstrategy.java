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

/**
 * Estrategias de aprendisaje
 *
 * @author Marcelo
 * @param <N>
 */
public interface LearningEstrategy<N extends Number> {

    Float[] floatBasicLearningSeries = {.0000000001f, 0.0000000003f, .0000000004f, .0000000005f, .0000000008f, .000000001f, 0.000000003f, .000000004f, .000000005f, .000000008f, .00000001f, 0.00000003f, .00000004f, .00000005f, .00000008f, .0000001f, 0.0000003f, .0000004f, .0000005f, .0000008f, .000001f, 0.000003f, .000004f, .000005f, .000008f, .00001f, 0.00003f, .00004f, .00005f, .00008f, .0001f, 0.0003f, .0004f, .0005f, .0008f, .001f, 0.003f, .004f, .005f, .008f, .01f, .03f, .04f, .05f, .08f, .1f, .3f, .4f, .5f, .8f};
    Double[] doubleBasicLearningSeries = {.0000000001, 0.0000000003, .0000000004, .0000000005, .0000000008, .000000001, 0.000000003, .000000004, .000000005, .000000008, .00000001, 0.00000003, .00000004, .00000005, .00000008, .0000001, 0.0000003, .0000004, .0000005, .0000008, .000001, 0.000003, .000004, .000005, .000008, .00001, 0.00003, .00004, .00005, .00008, .0001, 0.0003, .0004, .0005, .0008, .001, 0.003, .004, .005, .008, .01, .03, .04, .05, .08, .1, .3, .4, .5, .8};

    /**
     * obtiene nuevo factor de aprendisaje
     *
     * @param index the value of index - iteracion o tiempo
     * @param factor the value of factor - factor actualmente utilizado
     * @return the N
     */
    N updateFactor(int index, N factor);

}
