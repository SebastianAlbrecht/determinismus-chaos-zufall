# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:09:59 2021

@author: SEBASTIAN
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams
import time
import math

# Pfad zu meiner Kopie des FFMPEG Writers zum Speichern der Animation
rcParams['animation.ffmpeg_path'] = r'D:\Users\SEBASTIAN\.matplotlib\ffmpeg-2021-01-01-git-63505fc60a-full_build\bin\ffmpeg.exe'


# ---------- Berechnung Stoßprozesse ----------

def free_propagation(pos, vel, time_step, num_steps): 
    num_particles = pos.shape[0]
    start = np.tile(pos, num_steps).reshape((num_steps, num_particles))
    steps = time_step * np.tile(vel, num_steps).reshape((num_steps, num_particles))
    steps[0] = 0
       
    free_trajectories = start + np.cumsum(steps, axis = 0)
    
    return free_trajectories

    
def free_to_interacting(free_trajectories):   
    ones = np.ones_like(free_trajectories)
    
    max_right_excess = np.max(free_trajectories) - 1
    max_left_excess = 0 - np.min(free_trajectories)
    
    max_bounces = max(max_right_excess, max_left_excess) / 2.
    num_bounces = math.ceil(max_bounces)
        
    # hier ist der größte Zeitfresser (i.e. Code beschleunigen oder verhindern, dass es viele Reflektionen gibt)   
    for i in range(num_bounces):
        free_trajectories = np.abs( free_trajectories ) # left bounce
        free_trajectories = ones - np.abs( ones - free_trajectories ) # right bounce

    interacting_trajectories = np.sort(free_trajectories, axis = -1)
        
    return interacting_trajectories


def free_to_interacting_circle(free_trajectories):   
    num_steps = free_trajectories.shape[0]
    
    ones = np.ones_like(free_trajectories)
    zeros = np.zeros_like(free_trajectories)
    
    right_excess = np.where( (free_trajectories - ones) > 0, free_trajectories - ones, zeros)
    right_excess = np.ceil(right_excess).astype("int")
    left_excess = np.where( (zeros - free_trajectories) > 0, zeros - free_trajectories, zeros)
    left_excess = np.ceil(left_excess).astype("int")
    
    shifts = - ( np.sum(right_excess, axis = 1) - np.sum(left_excess, axis = 1) )
    
    free_trajectories = free_trajectories % 1.
    interacting_trajectories = np.sort(free_trajectories, axis = -1)
    for n in range(num_steps):
        interacting_trajectories[n] = np.roll(interacting_trajectories[n,:], shift = shifts[n])
        
    return interacting_trajectories #!!! hier muss nur der return value "shifts" entfernt werden


def free_to_interacting_no_walls(free_trajectories):
    interacting_trajectories = np.sort(free_trajectories, axis = -1)
    return interacting_trajectories



# ---------- Untersuchung mittlerer Punkt ----------
    
#!!!
def distribution_middle_particle_position(start_pos, start_vel, time_step, num_steps, boundary):
    
    num_reps = start_pos.shape[0]
    num_particles = start_pos.shape[1]
    middle_index = math.ceil(num_particles / 2)
    
    middle_trajectories = np.zeros((num_reps, num_steps))
    scale = np.linspace(1, num_steps, endpoint = True, num = num_steps)
    scale = (np.pi / (2 * scale))**(1/4)
    
    if boundary == "wall":
        for n in range(num_reps): 
            free_trajectories = free_propagation(start_pos[n], start_vel[n], time_step, num_steps)
            middle_trajectories[n,:] = free_to_interacting(free_trajectories)[:,middle_index]
            middle_trajectories[n,:] = scale * middle_trajectories[n,:]
    
    elif boundary == "periodic":
        for n in range(num_reps): 
            free_trajectories = free_propagation(start_pos[n], start_vel[n], time_step, num_steps)
            middle_trajectories[n,:] = free_to_interacting_circle(free_trajectories)[:,middle_index]
            middle_trajectories[n,:] = scale * middle_trajectories[n,:]
        
    return middle_trajectories
    

#!!!
def donsker_rescaling(interacting_trajectories, num_rescale_steps, num_time_steps):

    num_particles = interacting_trajectories.shape[1]
    middle_index = math.floor(num_particles / 2.)
    
    middle_trajectory = interacting_trajectories[:,middle_index]
    
    donsker_trajectory = np.zeros((num_rescale_steps,num_time_steps))

    for n in range(num_rescale_steps): 
        for step in range(num_time_steps): 
            t = step/num_time_steps
            donsker_trajectory[n,step] = 1./(np.sqrt(n+1)) * middle_trajectory[int(n*t)]
    
    return donsker_trajectory 



# ---------- Animation Stoßprozesse auf der Geraden ----------
    
#!!!
def anim_1D_colliding_particles(start_pos, start_vel, time_step = 0.1, num_steps = 200, boundary = "wall"):
    
    if boundary not in ["wall", "periodic", "free"]: boundary = "wall"
    
    free_trajectories = free_propagation(start_pos, start_vel, time_step, num_steps)
    if boundary == "wall": interacting_trajectories = free_to_interacting(free_trajectories)
    elif boundary == "periodic": interacting_trajectories = free_to_interacting_circle(free_trajectories)
    elif boundary == "free": interacting_trajectories = free_to_interacting_no_walls(free_trajectories)
    
    fig, axes = plt.subplots(1,2, figsize = (11,5))
    fig.subplots_adjust(wspace = 0.4)
        
    anim = animation.FuncAnimation(fig, update_colliding_particles_anim, interval=250, 
                                   frames = num_steps, 
                                   fargs = (interacting_trajectories, time_step, boundary), 
                                   repeat = False)
   
    return anim


#!!!
def update_colliding_particles_anim(i, interacting_trajectories, time_step, boundary):
    
    num_particles = interacting_trajectories.shape[1]    
    num_steps = interacting_trajectories.shape[0]
    middle_index = math.floor(num_particles / 2.)
    
    color_labels = np.full(shape=num_particles, fill_value=10,dtype=np.int)
    color_labels[middle_index]=0
    
    middle_trajectory = interacting_trajectories[:,middle_index]
    times = time_step * np.linspace(0, i, endpoint = False, num = i)
    
    fig = plt.gcf()
    
    [ax1, ax2] = fig.axes
    ax1.clear()
    ax2.clear()
    
    if boundary == "wall":
        ax1.set_xlim((-0.1, 1.1))
        ax1.plot([-0.01, 1.01], [1,1], c = "#c4c4c4", marker = "|", zorder = 0)
        ax1.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax1.get_yaxis().set_visible(False)
        ax1.scatter(interacting_trajectories[i], np.ones_like(interacting_trajectories[i]),
                    marker = "o", c = color_labels, cmap = "Set1")
        ax2.set_xlim(( 0.95 * np.min(middle_trajectory), 1.05 * np.max(middle_trajectory)))
    elif boundary == "periodic":
        ax1.set_xlim((-1.1,1.1))
        ax1.set_ylim((-1.1,1.1))
        ax1.set_aspect("equal")
        ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
        circle = plt.Circle(((0.,0.)), 1., fill=False, edgecolor='#c4c4c4', zorder = 0)
        ax1.add_patch(circle)
        ax1.scatter(np.sin(2*np.pi*interacting_trajectories[i]), np.cos(2*np.pi*interacting_trajectories[i]),
                    marker = "o", c = color_labels, cmap = "Set1")
        middle_trajectory = 2 * np.pi * middle_trajectory
        ax2.set_xticks([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
        ax2.set_xticklabels([r"$0$", r"$\frac{1}{2} \pi$", r"$\pi$", r"$\frac{3}{2} \pi$", r"$2 \pi$"])
        ax2.set_xlim(( np.min(middle_trajectory) - 1 , np.max(middle_trajectory) + 1))
    elif boundary == "free":
        x_lim_right = np.max(interacting_trajectories)
        x_lim_left = np.min(interacting_trajectories)
        x_lim = max(x_lim_right - 1, x_lim_left)
        ax1.set_xlim(-1.1*x_lim, 1 + 1.1*x_lim)
        ax1.plot([-1.1*x_lim, 1 + 1.1*x_lim], [1,1], c = "#c4c4c4", marker = "", zorder = 0)
        ax1.get_yaxis().set_visible(False)
        ax1.scatter(interacting_trajectories[i], np.ones_like(interacting_trajectories[i]),
                    marker = "o", c = color_labels, cmap = "Set1")
        ax2.set_xlim(( np.min(middle_trajectory) - 0.1, np.max(middle_trajectory) + 0.1))
    
    
    ax2.set_ylim((- time_step, time_step * (num_steps + 1)))  

    ax2.plot(middle_trajectory[0:i], times, c = "r")
    ax2.set_xlabel("Auslenkung des roten Teilchens")
    ax2.set_ylabel(r"$t$")
    
    return fig


#!!! die Funktionen anim_1D_colliding_particles_circle und update_colliding_particles_anim_circle 
# habe ich gelöscht, die braucht es jetzt natürlich nicht mehr ;)


# ---------- Animation Verteilung Position mittleres Teilchen ----------

#!!! (hier kommen aber gleich noch weitere Änderungen vermutlich)
def anim_distribution_middle_particle_pos(num_particles, speed, time_step, num_steps, num_reps, num_frames, 
                                          boundary = "wall"):
    
    if boundary not in ["wall", "periodic"]: boundary = "wall"
    
    middle_index = math.floor(num_particles / 2.)
    
    start_pos = np.random.random_sample((num_reps, num_particles))
    start_pos = np.sort(start_pos, axis = -1)
    
    offset = ( start_pos[:,middle_index] - 0.5 )
    offset = np.repeat(offset, num_reps).reshape((num_reps, num_particles))
    start_pos -= offset
    
    ones = np.ones_like(start_pos)
    start_pos = np.abs( start_pos ) # left bounce
    start_pos = ones - np.abs( ones - start_pos ) # right bounce
    
    start_vel = speed * (np.random.random_sample((num_reps, num_particles)) - 0.5)
    
    distribution_position = distribution_middle_particle_position(start_pos, start_vel, time_step, num_steps, 
                                                                  boundary)
    
    fig, ax = plt.subplots()
    
    steps = np.unique( np.geomspace(1, num_steps, num_frames, 
                                    endpoint = True, dtype = int) ) - 1

    anim = animation.FuncAnimation(fig, update_middle_particle_anim, interval=100, 
                                   frames = steps.size, 
                                   fargs = (distribution_position, steps, num_particles), 
                                   repeat = False)
   
    return anim


def update_middle_particle_anim(i, distribution_middle_particle_position, steps, num_particles):
    
    print(steps[i], end=" ")
        
    fig = plt.gcf()

    mean = np.mean( distribution_middle_particle_position[:,steps[i]] )

    ax = fig.axes[0]
    ax.clear()
    ax.set_xlim((-0.1,0.1))
    ax.hist( distribution_middle_particle_position[:,steps[i]] - mean, bins = 21 )
    ax.set_title(r"Verteilung von $(\frac{\pi}{2t})^{(1/4)} y_0(t)$")
    
    return 


# ---------- Animation Verteilung Rescaling2/Donsker ----------

def anim_middle_particle_donsker(start_pos, start_vel, time_step, num_steps, num_frames = None):
    
    if num_frames == None: num_frames = num_steps
    
    free_trajectories = free_propagation(start_pos, start_vel, time_step, num_steps)
    interacting_trajectories = free_to_interacting(free_trajectories)
    middle_trajectory_donsker = donsker_rescaling(interacting_trajectories, num_frames, num_frames)
    
    fig, ax = plt.subplots()
    
    anim = animation.FuncAnimation(fig, update_middle_particle_donsker_anim, interval=10, 
                                   frames = num_frames, 
                                   fargs = (middle_trajectory_donsker, num_frames), 
                                   repeat = False)
   
    return anim


def update_middle_particle_donsker_anim(i, middle_trajectory_donsker, anim_num_time_steps):
    
    print(i, end=" ")
    
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.clear()
    
    times = np.linspace(0,1,anim_num_time_steps)
    rescaled_middle_trajectory = middle_trajectory_donsker[i] - middle_trajectory_donsker[i,0]
    x_lim_right = max(rescaled_middle_trajectory)
    x_lim_left = min(rescaled_middle_trajectory)
    x_lim = max(- x_lim_left, x_lim_right)
    ax.set_xlim(- 1.05 * x_lim, 1.05* x_lim)
    ax.set_title(r"$Y_A$ für A = " + str(i))
    ax.plot(rescaled_middle_trajectory, times)
    
    return fig
      

# ---------- Hauptprogramm ----------

num_particles = 100
num_steps = 200  
time_step = 0.1
num_frames = 200
speed = 1

start_pos = np.random.random(num_particles)
start_vel = speed * (np.random.random(num_particles) - 0.5)

#anim = anim_middle_particle_donsker(start_pos, start_vel, time_step, num_steps)
#anim = anim_middle_particle(num_particles, speed, time_step, num_steps = 2000, num_reps = 500,
#                            num_frames = 100)

#anim = anim_1D_colliding_particles(start_pos, start_vel, boundary = "free")

writermp4 = animation.FFMpegWriter(fps = 25) 
#anim.save("name.mp4", writer=writermp4)

plt.show()
