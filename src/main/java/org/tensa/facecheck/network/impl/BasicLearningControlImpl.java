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
package org.tensa.facecheck.network.impl;

import java.util.Arrays;
import java.util.function.UnaryOperator;
import org.tensa.facecheck.network.LearningControl;

/**
 *
 * @author Marcelo
 */
public class BasicLearningControlImpl<N extends Number> implements LearningControl<N> {
    
    private final UnaryOperator<Integer> calculateIndex;
    private final N[] learningSeries;

    public BasicLearningControlImpl(UnaryOperator<Integer> calculateIndex, N[] learningSeries) {
        this.calculateIndex = calculateIndex;
        this.learningSeries = learningSeries;
    }
    
    /**
     *
     * @param index the value of index
     * @param factor the value of factor
     * @return the N
     */
    @Override
    public N updateFactor(int index, N factor) {
        int realIndex = Arrays.binarySearch(learningSeries, factor);
        
        Integer newDeltaIndex = calculateIndex.apply(index);
        int ultimateIndex = realIndex-newDeltaIndex;
        return learningSeries[(ultimateIndex>0?ultimateIndex<learningSeries.length?ultimateIndex:learningSeries.length-1:0)];
    }
    
    public static int cadaUnoAvanzaUnoCadaTresVuelvaUno(int i) {
        return i % 3 == 0 ? -1 : 1;
    }
    
    public static int cadaTresAvanzaUno(int i) {
        return i % 3 == 0 ? 1 : 0;
    }
    
    public static int cadaTresAvanzaUnoCadaNueveVuelvaUno(int i) {
        return i % 3 == 0 ? i % 9 == 0 ? -1 : 1 : 0;
    }
    
    public static int cadaNueveAvanzaUno(int i) {
        return i % 9 == 0 ? 1 : 0;
    }
}
