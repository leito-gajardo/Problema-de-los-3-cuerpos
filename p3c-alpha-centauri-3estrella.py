#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:23:28 2022

@author: leito_gajardo
"""

#SISTEMA DE 3 CUERPOS: ALPHA CENTAURI + 1 ESTRELLA

#MODULOS GENERALES
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation 
import scipy.integrate

#MODULOS CREADOS
from func_ecsmov_anim_p3o5c import p3o5c, update3
import constantes as ctes

#CONSTANTE DE GRAVITACION UNIVERSAL
G = ctes.CTE_GRAV_UNIVERSAL #Nm**2/kg**2

# OBS: BASTA CON CAMBIAR LOS VALORES DEL SIGUIENTE CUADRO PARA CREAR OTROS SISTEMAS DE 3 CUERPOS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#CANTIDADES PARA NORMALIZACION
m_norm = ctes.MASA_SOL #kg #masa del sol
r_norm = ctes.DISTANCIA_ALPHA_CENTAURI #m #distancia entre estrellas en alpha centauri
v_norm = ctes.VELOCIDAD_REL_TIERRA_SOL #m/s #velocidad relativa tierra-sol
t_norm = ctes.PERIODO_ORBITAL_ALPHA_CENTAURI #s #periodo orbital en alpha centauri

#CONSTANTES DE NORMALIZACION
K1=G*t_norm*m_norm/(r_norm**2*v_norm)
K2=v_norm*t_norm/r_norm

#MASAS NORMALIZADAS
m1 = ctes.MASA_ALPHA_CENTAURI_A/ctes.MASA_SOL #adimensional #Alpha Centauri A
m2 = ctes.MASA_ALPHA_CENTAURI_B/ctes.MASA_SOL #adimensional #Alpha Centauri B
m3 = 1.4*ctes.MASA_SOL/ctes.MASA_SOL #adimensional #estrella 3 (1.4 masa solar)
m = [m1,m2,m3] #arreglo de las 3 masas adimensionales

#POSICIONES INICIALES NORMALIZADAS
r1i=np.array([-0.5,0,0], dtype="float64") #distancia Alpha Centauri / 2 normalizada
r2i=np.array([0.5,0,0], dtype="float64") #distancia Alpha Centauri / 2 normalizadad
r3i=np.array([0,1,0], dtype="float64") #distancia tercera estrella

#VELOCIDADES INICIALES NORMALIZADAS
v1i=np.array([0.01,0.01,0], dtype="float64") #velociad relativa Alpha Centauri A
v2i=np.array([-0.05,0.0,-0.1], dtype="float64") #velociad relativa Alpha Centauri B
v3i=np.array([0,-0.01,0], dtype="float64") #velociad relativa tercera estrella

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ARREGLO DE CONDICIONES INICIALES DE LOS 3 CUERPOS DE 1D
cond_in=np.array([r1i,r2i,r3i,v1i,v2i,v3i]).flatten() #condiciones iniciales

#VALORES PARA ITERAR
N = 500 #cantidad de pasos o puntos en la simulacion
tmax = 16 #cantidad de orbitas tierra-sol (esta normalizado con t_norm)
t=np.linspace(0,16,N) #tiempo total

#SOLUCION DE LAS ECUACIONES DIFERENCIALES DE MOVIMIENTO DEL PROBLEMA DE LOS 3 CUERPOS 
p3c_sol = sp.integrate.odeint(p3o5c, cond_in, t, args=(G,m,K1,K2))

#ARREGLO DE LAS POSICIONES SOLUCION
r1_sol=p3c_sol[:,:3]
r2_sol=p3c_sol[:,3:6]
r3_sol=p3c_sol[:,6:9]

#FIGURA
fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

#DATOS QUE SE ITERARAN PARA SIMULAR
data = np.array([r1_sol, r2_sol, r3_sol])

#CURVAS TRAYECTORIA DE CADA CUERPO
line1, = ax.plot(data[0][0,0:1], data[0][0,1:2], data[0][0,2:3], lw=2, color="gold")
line2, = ax.plot(data[1][0,0:1], data[1][0,1:2], data[1][0,2:3], lw=2, color="orangered")
line3, = ax.plot(data[2][0,0:1], data[2][0,1:2], data[2][0,2:3], lw=2, color="purple")

#PUNTOS EN MOVIMIENTO DE CADA CUERPO
point1, = ax.plot(data[0][0,0:1], data[0][0,1:2], data[0][0,2:3], 'o', ms=13, color="gold", mec="red", label="Alpha Centauri A")
point2, = ax.plot(data[1][0,0:1], data[1][0,1:2], data[1][0,2:3], 'o', ms=9, color="orangered", mec="darkblue", label="Alpha Centauri B")
point3, = ax.plot(data[2][0,0:1], data[2][0,1:2], data[2][0,2:3], 'o', ms=11, color="indigo", mec="black", label="Tercera estrella")

#EJES
ax.set_xlim3d([-2.5,2.5])
ax.set_xlabel('$X$')

ax.set_ylim3d([-2.5,2.5])
ax.set_ylabel('$Y$')

ax.set_zlim3d([-2.5,2.5])
ax.set_zlabel('$Z$')

#TITULO Y LEYENDAS
ax.set_title("Sistema Alpha Centauri más tercera estrella",fontsize=14)
ax.legend(loc="upper left",fontsize=14)

#ANIMACION
ani = animation.FuncAnimation(fig, update3, frames=len(data[0]), fargs=(data, line1, line2, line3, point1, point2, point3), interval=100, blit=True)

#GUARDAR VIDEO DE ANIMACION EN MP4 USANDO FFMPEG WRITER
# writervideo = animation.FFMpegWriter(fps=60)
# ani.save('p3c-alpha-centauri-3estrella.mp4', writer=writervideo)
# plt.close()

#MOSTRAR ANIMACION
plt.show()