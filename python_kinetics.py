import numpy as np
from numpy.random import random
from itertools import combinations
from itertools import product

class Vector():
    '''2D vectors'''

    def __init__(self, i1,i2):
        '''Initialise vectors with x and y coordinates'''
        self.x = i1
        self.y = i2

    def __add__(self,other):
        '''Use + sign to implement vector addition'''
        return (Vector(self.x+other.x,self.y+other.y))

    def __sub__(self,other):
        '''Use - sign to implement vector "subtraction"'''
        return (Vector(self.x-other.x,self.y-other.y))

    def __mul__(self,number):
        '''Use * sign to multiply a vector by a scaler on the left'''
        return Vector(self.x*number,self.y*number)

    def __rmul__(self,number):
        '''Use * sign to multiply a vector by a scaler on the right'''
        return Vector(self.x*number,self.y*number)

    def __truediv__(self,number):
        '''Use / to multiply a vector by the inverse of a number'''
        return Vector(self.x/number,self.y/number)

    def __repr__(self):
        '''Represent a vector by a string of 2 coordinates separated by a space'''
        return '{x} {y}'.format(x=self.x, y=self.y)

    def copy(self):
        '''Create a new object which is a copy of the current.'''
        return Vector(self.x,self.y)

    def dot(self,other):
        '''Calculate the dot product between two 2D vectors'''
        return self.x*other.x + self.y*other.y

    def norm(self):
        '''Calculate the norm of the 2D vector'''
        return (self.x**2+self.y**2)**0.5
    
    def perpendicular_counterclockwise(self):
        return Vector(-self.y,self.x)

class Particle():
    def __init__(self, position, momentum, radius, mass, force):
        self.position=position
        self.momentum=momentum
        self.radius=radius
        self.mass=mass
        self.force=force

    def velocity(self):
        return self.momentum/self.mass

    def copy(self):
        return Particle(self.position,self.momentum,self.radius,self.mass,self.force)

    def overlap(self, other_particle):
        distance=(other_particle.position-self.position).norm()
        return distance<(self.radius+other_particle.radius)

class Track():
    def __init__(self, array):
        self.geometry=array#array of vectors
        
    def overlap(self, particle):
        distance=(particle.position-self.geometry[0]).norm()
        index=0
        for i,pos in enumerate(self.geometry,start=1):
            curr_distance=(particle.position-pos).norm()
            if curr_distance<distance:
                distance=curr_distance
                index=i
        return distance<particle.radius,index

class Simulation():
    def __init__(self, particles, tracks, box_length, dt):
        self.particles=particles
        self.tracks=tracks
        self.box_length=box_length
        self.dt=dt
        self.trajectory=[]
        if len(self.particles)>0: self.record_state()
        
    def apply_particle_collision(self,particle1,particle2):
        dot_p=(particle2.momentum-particle1.momentum).dot(particle2.position-particle1.position)<0
        if particle1.overlap(particle2) and dot_p==True:
            normalised_collision_axis=(particle2.position-particle1.position)/(particle2.position-particle1.position).norm()
            p1_proj_coll=normalised_collision_axis*particle1.momentum.dot(normalised_collision_axis)
            p2_proj_coll=normalised_collision_axis*particle2.momentum.dot(normalised_collision_axis)
            p1_proj_collp=particle1.momentum-p1_proj_coll
            p2_proj_collp=particle2.momentum-p2_proj_coll
            collframep1=((particle1.mass-particle2.mass)*p1_proj_coll+2*particle1.mass*p2_proj_coll)/(particle1.mass+particle2.mass)+p1_proj_collp
            collframep2=((particle2.mass-particle1.mass)*p2_proj_coll+2*particle2.mass*p1_proj_coll)/(particle2.mass+particle1.mass)+p2_proj_collp
            
            particle1.momentum=collframep1
            particle2.momentum=collframep2
            
        
    def apply_box_collisions(self,particle):#simplify
        if (particle.position.x > self.box_length and particle.momentum.x>0) or (particle.position.x < 0 and particle.momentum.x<0):
            particle.momentum.x=-particle.momentum.x
        if (particle.position.y > self.box_length and particle.momentum.y>0) or (particle.position.y < 0 and particle.momentum.y<0):
            particle.momentum.y=-particle.momentum.y
        
    def apply_ramp_collisions(self,particle,track):
        overlap,ramp_index=track.overlap(particle)
        if overlap:
            if ramp_index==0:
                linear_aprox=track.geometry[1]-track.geometry[0]
            elif ramp_index==len(track.geometry)-1:
                linear_aprox=track.geometry[-1]-track.geometry[-2]
            else:
                linear_aprox=track.geometry[ramp_index+1]-track.geometry[ramp_index-1]
            normalised_collision_axis=linear_aprox.perpendicular_counterclockwise()/linear_aprox.perpendicular_counterclockwise().norm()
            proj_coll=normalised_collision_axis*particle.momentum.dot(normalised_collision_axis)
            proj_collp=particle.momentum-proj_coll
            particle.momentum=-1*proj_coll+proj_collp
        
    def step(self):
        for p1,p2 in combinations(self.particles,2):
            self.apply_particle_collision(p1,p2)
        for p,t in product(self.particles,self.tracks):
            self.apply_ramp_collisions(p,t)
        for p in self.particles:
            self.apply_box_collisions(p)
            p.force=Vector(0,-9.8*p.mass)
            p.position=p.position+p.velocity()*self.dt+0.5*(p.force/p.mass)*self.dt**2
            p.momentum=p.momentum+p.force*self.dt
        self.record_state()

    def record_state(self):
        state=[]
        for p in self.particles:
            state.append(p.copy())
        self.trajectory.append(state)
        
    def E_analysis(self):
        kinetic_energy=[]
        potential_energy=[]
        for states in self.trajectory:
            ke_state_energy=0
            pe_state_energy=0
            for particle in states:
                ke_state_energy+=(particle.momentum.norm()**2)/(2*particle.mass)
                pe_state_energy+=particle.mass*9.8*particle.position.y
            kinetic_energy+=[ke_state_energy]
            potential_energy+=[pe_state_energy]
        time=np.arange(0,len(kinetic_energy)*self.dt,self.dt)
        return time,kinetic_energy,potential_energy