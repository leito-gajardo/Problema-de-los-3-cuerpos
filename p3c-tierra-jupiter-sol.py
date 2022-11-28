#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 22:58:08 2022

@author: leito_gajardo
"""

#SISTEMA DE 3 CUERPOS: TIERRA-JUPITER-SOL

#MODULOS GENERALES
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation 
import scipy.integrate

#MODULOS CREADOS
from func_ecsmov_anim_p3o5c import p3o5c, update3 #funcion con ecuaciones de movimiento (p3o5) y funcion update para animar (ambas para 3 o 5 cuerpos)
import constantes as ctes #valores reales constantes a utilizar

#CONSTANTE DE GRAVITACION UNIVERSAL
G = ctes.CTE_GRAV_UNIVERSAL #Nm**2/kg**2

# OBS: BASTA CON CAMBIAR LOS VALORES DEL SIGUIENTE CUADRO PARA CREAR OTROS SISTEMAS DE 3 CUERPOS
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
m1 = ctes.MASA_SOL/ctes.MASA_SOL #asimensional #Sol
m2 = ctes.MASA_TIERRA/ctes.MASA_SOL #adimensional #Tierra
m3 = ctes.MASA_JUPITER/ctes.MASA_SOL #adimensional #Jupiter
m = [m1,m2,m3] #arreglo de masas

#POSICIONES INICIALES NORMALIZADAS
r1i = np.array([0.,0.,0.], dtype="float64") #Sol en origen
r2i = np.array([1,0.,0.], dtype="float64") #distancia tierra-sol (1UA) normalizada
r3i = np.array([-5.0,0.,0.], dtype="float64") #distancia jupiter-sol (5UA) normalizada

#VELOCIDADES INICIALES NORMALIZADAS
v1i = np.array([0.,0.,0.], dtype="float64") #velocidad relativa sol (reposo) normalizada
v2i = np.array([0.,1.,0.], dtype="float64") #velocidad relativa tierra normalizada
v3i = np.array([0.,-0.45,0.], dtype="float64") #velocidad relativa jupiter normalizada

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ARREGLO DE CONDICIONES INICIALES DE LOS 3 CUERPOS DE 1D
cond_in = np.array([r1i, r2i, r3i, v1i, v2i, v3i]) #condiciones iniciales
cond_in = cond_in.flatten()

#VALORES PARA ITERAR
N = 500 #cantidad de pasos o puntos en simulacion
tmax = 16 #cantidad de orbitas tierra-sol (esta normalizado con t_norm) 
t=np.linspace(0, tmax, N) #tiempo total

#SOLUCION DE LAS ECUACIONES DIFERENCIALES DE MOVIMIENTO DEL PROBLEMA DE LOS 3 CUERPOS 
p3c_sol = sp.integrate.odeint(p3o5c, cond_in, t, args=(G,m,K1,K2))

#ARREGLOS DE LAS POSICIONES SOLUCION
r1_sol=p3c_sol[:,:3]
r2_sol=p3c_sol[:,3:6]
r3_sol=p3c_sol[:,6:9]

#FIGURA 3D
fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

#DATOS QUE SE ITERARAN PARA SIMULAR
data = np.array([r1_sol, r2_sol, r3_sol])

#CURVAS TRAYECTORIA DE CADA CUERPO
line1, = ax.plot(data[0][0,0:1], data[0][0,1:2], data[0][0,2:3], lw=2, color="gold")
line2, = ax.plot(data[1][0,0:1], data[1][0,1:2], data[1][0,2:3], lw=2, color="tab:blue")
line3, = ax.plot(data[2][0,0:1], data[2][0,1:2], data[2][0,2:3], lw=2, color="brown")

#PUNTOS EN MOVIMIENTO DE CADA CUERPO
point1, = ax.plot(data[0][0,0:1], data[0][0,1:2], data[0][0,2:3], 'o', ms=15, color="gold", mec="red", label="Sol")
point2, = ax.plot(data[1][0,0:1], data[1][0,1:2], data[1][0,2:3], 'o', ms=8, color="green", mec="blue", label="Tierra")
point3, = ax.plot(data[2][0,0:1], data[2][0,1:2], data[2][0,2:3], 'o', ms=10, color="brown", mec="black", label="Jupiter")

#EJES
ax.set_xlim3d([-4.5,4.5])
ax.set_xlabel('$X$')

ax.set_ylim3d([-4.5,4.5])
ax.set_ylabel('$Y$')

ax.set_zlim3d([-2.5,2.5])
ax.set_zlabel('$Z$')

#TITULO Y LEYENDAS
ax.set_title("Órbita de planetas en un sistema solar de 3 cuerpos", fontsize=16)
ax.legend(loc="upper left",fontsize=14)

#ANIMACION
ani = animation.FuncAnimation(fig, update3, frames=N, fargs=(data, line1, line2, line3, point1, point2, point3), interval=5, blit=True)

#GUARDAR VIDEO DE ANIMACION EN MP4 USANDO FFMPEG WRITER
# writervideo = animation.FFMpegWriter(fps=60)
# ani.save('p3c-tierra-jupiter-sol.mp4', writer=writervideo)
# plt.close()

#MOSTRAR ANIMACION
plt.show()
