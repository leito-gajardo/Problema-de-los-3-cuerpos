#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:01:38 2022

@author: leito_gajardo
"""

import numpy as np

def p3o5c(w, t, G, m, K1, K2):
    ''' Función que contiene las ecuaciones de movimiento (provenientes de la
    segunda ley de Newton con su Ley de Gravitacion Universal) para el 
    problema de los 3 y 5 cuerpos, donde se definen las aceleraciones iniciales
    de cada cuerpo en base al arreglo de condiciones iniciales de posición y 
    velocidad de cada cuerpo.
    
    Entrega un arreglo que contiene d/dt([x,v])=(v,a) (con x, v y a vectores)
    de cada cuerpo, para el arreglo de valores que se le entregue.
    
    Se le entrega: un arreglo 1D de condiciones iniciales de los 3 o 5 
    cuerpos de la forma w = array([r1i, r2i, r3i, v1i, v2i, v3i]) (pueden ser 
    5 r y 5 v para los 5 cuerpos) con rji=array([xji, yji, zji]) 
    y vji = array([vji_x, vji_y, vji_z]); un arreglo que represente el paso 
    del tiempo con N valores desde un tiempo inicial a uno final; un arreglo 
    que contenga las masas de la forma m = [m1, m2, m3] (pueden ser 5 m); y 
    por último las constantes de normalizacion K1 y K2.
    
    Esta funcion se le entrega como argumento a la función 
    scipy.integrate.odeint para que haga el proceso de integración de las 
    ecuaciones diferenciales que se definen en p3o5c.'''
    
    if len(w) == 18:
        r1 = w[:3]
        r2 = w[3:6]
        r3 = w[6:9]
        v1 = w[9:12]
        v2 = w[12:15]
        v3 = w[15:18]
        
        r12 = np.linalg.norm(r2-r1)
        r13 = np.linalg.norm(r3-r1)
        r23 = np.linalg.norm(r3-r2)
        
        a1 = K1*m[1]*(r2-r1)/r12**3 + K1*m[2]*(r3-r1)/r13**3
        a2 = K1*m[0]*(r1-r2)/r12**3 + K1*m[2]*(r3-r2)/r23**3
        a3 = K1*m[0]*(r1-r3)/r13**3 + K1*m[1]*(r2-r3)/r23**3
        v11 = K2*v1
        v22 = K2*v2
        v33 = K2*v3
        
        dr_dv = np.array([v11, v22, v33, a1, a2, a3]).flatten()
    
    if len(w) == 30:
        r1 = w[:3]
        r2 = w[3:6]
        r3 = w[6:9]
        r4 = w[9:12]
        r5 = w[12:15]
        v1 = w[15:18]
        v2 = w[18:21]
        v3 = w[21:24]
        v4 = w[24:27]
        v5 = w[27:30]
        
        r12 = np.linalg.norm(r2-r1)
        r13 = np.linalg.norm(r3-r1)
        r14 = np.linalg.norm(r4-r1)
        r15 = np.linalg.norm(r5-r1)
        r23 = np.linalg.norm(r3-r2)
        r24 = np.linalg.norm(r4-r2)
        r25 = np.linalg.norm(r5-r2)
        r34 = np.linalg.norm(r4-r3)
        r35 = np.linalg.norm(r5-r3)
        r45 = np.linalg.norm(r5-r4)
        
        a1 = K1*m[1]*(r2-r1)/r12**3 + K1*m[2]*(r3-r1)/r13**3 + K1*m[3]*(r4-r1)/r14**3 + K1*m[4]*(r5-r1)/r15**3
        a2 = K1*m[0]*(r1-r2)/r12**3 + K1*m[2]*(r3-r2)/r23**3 + K1*m[3]*(r4-r2)/r24**3 + K1*m[4]*(r5-r2)/r25**3
        a3 = K1*m[0]*(r1-r3)/r13**3 + K1*m[1]*(r2-r3)/r23**3 + K1*m[3]*(r4-r3)/r34**3 + K1*m[4]*(r5-r3)/r35**3
        a4 = K1*m[0]*(r1-r4)/r14**3 + K1*m[1]*(r2-r4)/r24**3 + K1*m[2]*(r3-r4)/r34**3 + K1*m[4]*(r5-r4)/r45**3
        a5 = K1*m[0]*(r1-r5)/r15**3 + K1*m[1]*(r2-r5)/r25**3 + K1*m[2]*(r3-r5)/r35**3 + K1*m[3]*(r4-r5)/r45**3
        v11 = K2*v1
        v22 = K2*v2
        v33 = K2*v3
        v44 = K2*v4
        v55 = K2*v5
        
        dr_dv = np.array([v11, v22, v33, v44, v55, a1, a2, a3, a4, a5]).flatten()
    return dr_dv


def update3(i, data, line1, line2, line3, point1, point2, point3):

    line1.set_data(data[0][:i, :2].T)
    line1.set_3d_properties(data[0][:i, 2:3].T[0])

    line2.set_data(data[1][:i, :2].T)
    line2.set_3d_properties(data[1][:i, 2:3].T[0])

    line3.set_data(data[2][:i, :2].T)
    line3.set_3d_properties(data[2][:i, 2:3].T[0])

    point1.set_data(data[0][i, :2].T)
    point1.set_3d_properties(data[0][i, 2:3].T[0])

    point2.set_data(data[1][i, :2].T)
    point2.set_3d_properties(data[1][i, 2:3].T[0])

    point3.set_data(data[2][i, :2].T)
    point3.set_3d_properties(data[2][i, 2:3].T[0])

    return line1, line2, line3, point1, point2, point3

def update5(i, data, line1, line2, line3, line4, line5, point1, point2, point3, point4, point5):

    line1.set_data(data[0][:i, :2].T)
    line1.set_3d_properties(data[0][:i, 2:3].T[0])

    line2.set_data(data[1][:i, :2].T)
    line2.set_3d_properties(data[1][:i, 2:3].T[0])

    line3.set_data(data[2][:i, :2].T)
    line3.set_3d_properties(data[2][:i, 2:3].T[0])

    line4.set_data(data[3][:i, :2].T)
    line4.set_3d_properties(data[3][:i, 2:3].T[0])

    line5.set_data(data[4][:i, :2].T)
    line5.set_3d_properties(data[4][:i, 2:3].T[0])

    point1.set_data(data[0][i, :2].T)
    point1.set_3d_properties(data[0][i, 2:3].T[0])

    point2.set_data(data[1][i, :2].T)
    point2.set_3d_properties(data[1][i, 2:3].T[0])

    point3.set_data(data[2][i, :2].T)
    point3.set_3d_properties(data[2][i, 2:3].T[0])

    point4.set_data(data[3][i, :2].T)
    point4.set_3d_properties(data[3][i, 2:3].T[0])

    point5.set_data(data[4][i, :2].T)
    point5.set_3d_properties(data[4][i, 2:3].T[0])

    return line1, line2, line3, line4, line5, point1, point2, point3, point4, point5
