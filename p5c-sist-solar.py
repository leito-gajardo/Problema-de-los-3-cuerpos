#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:11:00 2022

@author: leito_gajardo
"""

#SISTEMA DE 5 CUERPOS: VENUS-TIERRA-JUPITER-URANO-SOL

#MODULOS GENERALES
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation 
import scipy.integrate

#MODULOS CREADOS
from func_ecsmov_anim_p3o5c import p3o5c, update5 #funcion con ecuaciones de movimiento (p3o5) y funcion update para animar (ambas para 3 o 5 cuerpos)
import constantes as ctes #valores reales constantes a utilizar

#CONSTANTE DE GRAVITACION UNIVERSAL
G = ctes.CTE_GRAV_UNIVERSAL #Nm**2/kg**2

# OBS: BASTA CON CAMBIAR LOS VALORES DEL SIGUIENTE CUADRO PARA CREAR OTROS SISTEMAS DE 5 CUERPOS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#CANTIDADES PARA NORMALIZACION
m_norm = ctes.MASA_SOL #kg #masa del sol
r_norm = sp.constants.astronomical_unit #m #distancia tierra-sol (1UA)
v_norm = ctes.VELOCIDAD_REL_TIERRA_SOL #m/s #velocidad relativa tierra-sol
t_norm = ctes.PERIODO_ORBITAL_TIERRA_SOL #s #periodo orbital tierra-sol

#CONSTANTES DE NORMALIZACION
K1 = G*t_norm*m_norm / (r_norm**2*v_norm)
K2 = v_norm*t_norm / r_norm

#MASAS NORMALIZADAS
m1 = ctes.MASA_SOL/ctes.MASA_SOL #adimensional #Sol
m2 = ctes.MASA_VENUS/ctes.MASA_SOL #dimensional #Venus
m3 = ctes.MASA_TIERRA/ctes.MASA_SOL #adimensional #Tierra
m4 = ctes.MASA_JUPITER/ctes.MASA_SOL #adimensional #Jupiter
m5 = ctes.MASA_URANO/ctes.MASA_SOL #adimensional #Urano
m = [m1,m2,m3,m4,m5] #arreglo de masas

#POSICIONES INICIALES NORMALIZADAS
r1i = np.array([0.,0.,0.], dtype="float64") #Sol en origen
r2i = np.array([0.,0.72,0.], dtype="float64") #distancia venus-sol (0.72UA)
r3i = np.array([1.,0.,0.], dtype="float64") #distancia tierra-sol (1UA) normalizada
r4i = np.array([-5.2,0.,0.], dtype="float64") #distancia jupiter-sol (5.2UA) normalizada
r5i = np.array([0.,-19.22,0.], dtype="float64") #distancia urano-sol (19.22UA) normalizada

#VELOCIDADES INICIALES NORMALIZADAS
v1i = np.array([0.,0.,0.], dtype="float64") #velocidad relativa sol (reposo) normalizada
v2i = np.array([1.1,0.,0.], dtype="float64") #velocidad relativa venus normalizada
v3i = np.array([0.,1.,0.], dtype="float64") #velocidad relativa tierra normalizada
v4i = np.array([0.,-0.45,0.], dtype="float64") #velocidad relativa jupiter normalizada
v5i = np.array([-0.2,0.,0.], dtype="float64") #velocidad relativa urano normalizada

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ARREGLO DE CONDICIONES INICIALES DE LOS 5 CUERPOS DE 1D
cond_in = np.array([r1i, r2i, r3i, r4i, r5i, v1i, v2i, v3i, v4i, v5i]) #condiciones iniciales
cond_in = cond_in.flatten()

#VALORES PARA ITERAR
N = 800 #cantidad de pasos o puntos en simulacion
tmax = 64 #cantidad de orbitas tierra-sol (esta normalizado con t_norm) 
t=np.linspace(0, tmax, N) #tiempo total

#SOLUCION DE LAS ECUACIONES DIFERENCIALES DE MOVIMIENTO DEL PROBLEMA DE LOS 3 CUERPOS 
p5c_sol = sp.integrate.odeint(p3o5c, cond_in, t, args=(G,m,K1,K2))

#ARREGLOS DE LAS POSICIONES SOLUCION
r1_sol = p5c_sol[:,:3]
r2_sol = p5c_sol[:,3:6]
r3_sol = p5c_sol[:,6:9]
r4_sol = p5c_sol[:,9:12]
r5_sol = p5c_sol[:,12:15]

#FIGURA 3D
fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

#DATOS QUE SE ITERARAN PARA SIMULAR
data = np.array([r1_sol, r2_sol, r3_sol, r4_sol, r5_sol])

#CURVAS TRAYECTORIA DE CADA CUERPO
line1, = ax.plot(data[0][0,0:1], data[0][0,1:2], data[0][0,2:3], lw=2, color="gold")
line2, = ax.plot(data[1][0,0:1], data[1][0,1:2], data[1][0,2:3], lw=2, color="orange")
line3, = ax.plot(data[2][0,0:1], data[2][0,1:2], data[2][0,2:3], lw=2, color="green")
line4, = ax.plot(data[3][0,0:1], data[3][0,1:2], data[3][0,2:3], lw=2, color="brown")
line5, = ax.plot(data[4][0,0:1], data[4][0,1:2], data[4][0,2:3], lw=2, color="royalblue")

#PUNTOS EN MOVIMIENTO DE CADA CUERPO
point1, = ax.plot(data[0][0,0:1], data[0][0,1:2], data[0][0,2:3], 'o', ms=15, color="gold", mec="red", label="Sol")
point2, = ax.plot(data[1][0,0:1], data[1][0,1:2], data[1][0,2:3], 'o', ms=7, color="orange", mec="darkorange", label="Venus")
point3, = ax.plot(data[2][0,0:1], data[2][0,1:2], data[2][0,2:3], 'o', ms=8, color="green", mec="blue", label="Tierra")
point4, = ax.plot(data[3][0,0:1], data[3][0,1:2], data[3][0,2:3], 'o', ms=11, color="brown", mec="black", label="Jupiter")
point5, = ax.plot(data[2][0,0:1], data[2][0,1:2], data[2][0,2:3], 'o', ms=9, color="steelblue", mec="royalblue", label="Urano")

#EJES
ax.set_xlim3d([-20.5,20.5])
ax.set_xlabel('$X$')

ax.set_ylim3d([-20.5,20.5])
ax.set_ylabel('$Y$')

ax.set_zlim3d([-2.5,2.5])
ax.set_zlabel('$Z$')

#TITULO Y LEYENDAS
ax.set_title("Órbita de planetas en un sistema solar de 5 cuerpos", fontsize=16)
ax.legend(loc="upper left",fontsize=14)

#ANIMACION
ani = animation.FuncAnimation(fig, update5, frames=N, fargs=(data, line1, line2, line3, line4, line5, point1, point2, point3, point4, point5), interval=8, blit=True)

#GUARDAR VIDEO DE ANIMACION EN MP4 USANDO FFMPEG WRITER
# writervideo = animation.FFMpegWriter(fps=60)
# ani.save('p5c-sist-solar-zoom.mp4', writer=writervideo)
# plt.close()

#MOSTRAR ANIMACION
plt.show()
