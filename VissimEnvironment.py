# -*- coding: utf-8 -*-
"""
VISSIM Environment Interface Module

This module provides an interface to the VISSIM traffic simulation software for reinforcement learning.
It handles communication between the RL algorithm and VISSIM, including:
- Opening and controlling VISSIM simulation
- Tracking vehicle states
- Transmitting action commands
- Computing state-action rewards

@author: Frank Yan
"""
from __future__ import print_function
import os
# COM-Server for VISSIM communication
import win32com.client as com
import math
import numpy as np


# Standard lane width in meters
lanewidth = 3.75

class Vehicle:
    """
    Class representing a vehicle in the VISSIM environment
    
    Tracks position, speed, and adjacent vehicle relationships
    """
    def __init__(self, veh):
        """Initialize vehicle object with VISSIM vehicle data"""
        self.id     = veh.AttValue('No')
        self.length = veh.AttValue('Length')
        self.lonpos = veh.AttValue('Pos')
        self.lane = int(veh.AttValue('Lane').split('-')[1])
        self.latpos = (self.lane-1)*lanewidth + veh.AttValue('PosLat')*lanewidth
        self.speed = veh.AttValue('Speed')/3.6  # Convert km/h to m/s
        # Track IDs of adjacent vehicles (F=Front, R=Rear, L=Left, R=Right)
        self.FVid =  veh.AttValue('FVid')   # Front vehicle ID
        self.RVid =  veh.AttValue('RVid')   # Rear vehicle ID
        self.LFVid = veh.AttValue('LFVid')  # Left-front vehicle ID
        self.LRVid = veh.AttValue('LRVid')  # Left-rear vehicle ID
        self.RFVid = veh.AttValue('RFVid')  # Right-front vehicle ID
        self.RRVid = veh.AttValue('RRVid')  # Right-rear vehicle ID
        self.angle = veh.AttValue('Angle')
        
        self.initial_speed = self.speed
        self.adjenctvehicles=[self.FVid, self.RVid, self.LFVid, self.LRVid, self.RFVid, self.RRVid]
        self.traj = np.array([self.lonpos, self.latpos])
    
    def update(self, veh):
        """Update vehicle state with new VISSIM vehicle data"""
        self.lonpos = veh.AttValue('Pos')
        self.lane = int(veh.AttValue('Lane').split('-')[1])
        self.latpos = (self.lane-1)*lanewidth + veh.AttValue('PosLat')*lanewidth
        self.speed = veh.AttValue('Speed')/3.6  # Convert km/h to m/s
        # Update adjacent vehicle IDs
        self.FVid =  veh.AttValue('FVid')
        self.RVid =  veh.AttValue('RVid')
        self.LFVid = veh.AttValue('LFVid')
        self.LRVid = veh.AttValue('LRVid')
        self.RFVid = veh.AttValue('RFVid')
        self.RRVid = veh.AttValue('RRVid')
        self.angle = veh.AttValue('Angle')
        
        self.adjenctvehicles = [self.FVid, self.RVid, self.LFVid, self.LRVid, self.RFVid, self.RRVid]
        # Add new position to trajectory history
        self.traj = np.append(self.traj, [self.lonpos, self.latpos])


class VisEnv:
    """
    VISSIM Environment class for reinforcement learning
    
    Handles simulation control, state observation, and reward computation
    """
    def __init__(self, filedir, inpx, layx):
        """Initialize VISSIM environment with network and layout files"""
        # Connect to VISSIM COM server
        self.Vissim = com.gencache.EnsureDispatch("Vissim.Vissim.22")  # Vissim 2022
        self.inpx = os.path.join(filedir, inpx)  # Network file path
        self.layx = os.path.join(filedir, layx)  # Layout file path
        self.actiontext = os.path.join(filedir, 'action.txt')  # Communication file for actions
        
        # Environment configuration
        self.statenum = 20  # State dimension [ego lateral position, ego speed, surrounding vehicles' relative positions and speeds]
        self.actionnum = 2  # Action dimension [acceleration, steering angle]
        self.max_episode_length = 149  # Max steps per episode (corner_case=79, highway=149)
        self.maxangle = 0.20  # Maximum steering angle
        self.maxacc = 2  # Maximum acceleration
        self.timestep = 0.1  # Simulation time step
        self.load_network()

    def load_network(self):
        """Load VISSIM network and layout files"""
        flag_read_additionally = False
        self.Vissim.LoadNet(self.inpx, flag_read_additionally)
        self.Vissim.LoadLayout(self.layx)
        
    def reset(self, seed):
        """
        Reset simulation environment
        
        Args:
            seed: Random seed for simulation
            
        Returns:
            Initial state observation
        """
        # Configure simulation settings
        End_of_simulation = 196  # Simulation end time in seconds (corner_case: 9, highway: 196)
        break_of_simulation = 180  # Simulation break time (for highway test)
        Randseed = seed
        
        # Apply simulation settings
        self.Vissim.Simulation.SetAttValue('UseMaxSimSpeed', True)
        self.Vissim.Simulation.SetAttValue('RandSeed', Randseed)
        self.Vissim.Simulation.SetAttValue('SimPeriod', End_of_simulation)
        self.Vissim.Simulation.SetAttValue('SimBreakAt', break_of_simulation)
        self.Vissim.Simulation.RunContinuous()
        self.setaction(0, 0)  # Initialize with zero actions
        self.Vissim.Simulation.RunSingleStep()
        
        # Get ego vehicle (vehicle type 630)
        self.egovehicle = Vehicle(self.Vissim.Net.Vehicles.GetFilteredSet('[VehType]="630"').Iterator.Item)
        self.s_initial = self.get_state()
        
        return self.s_initial
    
    def step(self, action):
        """
        Execute one simulation step with the given action
        
        Args:
            action: Action to execute [acceleration, steering angle]
            
        Returns:
            next_state: Next state observation
            reward: Reward for the action
            done: Whether the episode has ended
        """
        # Scale and apply actions
        self.setaction(action[0]*self.maxacc, action[1]*self.maxangle)
        self.Vissim.Simulation.RunSingleStep()  # Run one simulation step
        self.egovehicle.update(self.Vissim.Net.Vehicles.ItemByKey(self.egovehicle.id))  # Update ego vehicle state
        s_next = self.get_state()  # Get next state
        reward, done = self.reward_compute(s_next)  # Compute reward and episode termination
        
        if done:
            self.Vissim.Simulation.Stop()

        return s_next, reward, done
    
    def get_state(self):
        """
        Get current state observation
        
        Returns:
            state: Current state vector [ego position, ego speed, surrounding vehicles' relative states]
        """
        state = []

        # Add ego vehicle state
        state.append(self.egovehicle.latpos)
        state.append(self.egovehicle.speed)
        
        # Add surrounding vehicles' states
        for index, vehicleid in enumerate(self.egovehicle.adjenctvehicles):
            if vehicleid < 0:
                # No vehicle present in this position - create virtual vehicle
                lane_no = index // 2  # Lane number: 0=ego lane, 1=left lane, 2=right lane
                position_no = index % 2  # Position: 0=front, 1=rear
                
                # Set appropriate relative positions for virtual vehicles based on lane configuration
                if self.egovehicle.lane == 2:
                    # Middle lane
                    if position_no == 0:
                        rel_lonpos = 150  # Far ahead
                    else:
                        rel_lonpos = -150  # Far behind
                elif self.egovehicle.lane == 1:
                    # Leftmost lane
                    if position_no == 0:
                        if lane_no == 2:                            
                            rel_lonpos = 1.5
                        else:
                            rel_lonpos = 150
                    else:
                        if lane_no == 2:
                            rel_lonpos = -1.5
                        else:
                            rel_lonpos = -150
                elif self.egovehicle.lane == 3:
                    # Rightmost lane
                    if position_no == 0:
                        if lane_no == 1:                            
                            rel_lonpos = 1.5
                        else:
                            rel_lonpos = 150
                    else:
                        if lane_no == 1:
                            rel_lonpos = -1.5
                        else:
                            rel_lonpos = -150
                
                # Set lateral positions relative to ego vehicle
                if lane_no == 0:
                    rel_latpos = 0  # Same lane
                elif lane_no == 1:
                    if self.egovehicle.lane == 3:
                        rel_latpos = 2
                    else:
                        rel_latpos = lanewidth
                elif lane_no == 2:
                    if self.egovehicle.lane == 1:
                        rel_latpos = -2
                    else:
                        rel_latpos = -lanewidth
                
                rel_speed = 0  # Virtual vehicles have same speed as ego
            else:
                # Real vehicle present - get its relative position and speed
                adj_vehicle = self.Vissim.Net.Vehicles.ItemByKey(vehicleid)
                lane = int(adj_vehicle.AttValue('Lane').split('-')[1])
                rel_lonpos = adj_vehicle.AttValue('Pos') - self.egovehicle.lonpos 
                rel_latpos = (lane-1)*lanewidth + adj_vehicle.AttValue('PosLat')*lanewidth - self.egovehicle.latpos
                rel_speed = adj_vehicle.AttValue('Speed')/3.6 - self.egovehicle.speed
            
            # Add relative position and speed to state
            state.append(rel_lonpos)
            state.append(rel_latpos)
            state.append(rel_speed)
            
        return state
        
    def setaction(self, desiredacc, desiredangle):
        """
        Send action commands to VISSIM through text file
        
        Args:
            desiredacc: Desired acceleration
            desiredangle: Desired steering angle
        """
        with open(self.actiontext, 'w') as file:
            file.write(f'{desiredacc},{desiredangle}')
            
    def TimeGapFront(self, state):
        """
        Calculate time gap to front vehicle
        
        Args:
            state: Current state vector
            
        Returns:
            front_dis: Distance to front vehicle
            front_gap: Time gap to front vehicle (distance/speed)
        """
        front_dis = state[2]
        if self.egovehicle.FVid < 0:
            front_gap = 100  # No front vehicle
        else:
            # Time gap = (distance - front vehicle length) / ego speed
            front_gap = (front_dis - self.Vissim.Net.Vehicles.ItemByKey(self.egovehicle.FVid).AttValue('Length')) / state[1]
        return front_dis, front_gap
        
    def reward_compute(self, state_next):
        """
        Compute reward for current state-action pair
        
        The reward considers:
        - Mobility: Speed reward
        - Comfort: Jerk and steering penalties
        - Safety: Gap, collision, and lane departure penalties
        
        Args:
            state_next: Next state after taking action
            
        Returns:
            reward: Computed reward value
            done: Whether episode is terminated (collision or end of simulation)
        """
        done = False
        # Reward weights [speed, jerk, steering, gap, collision, lateral]
        theta = [1.5, -0.05, -2.0, -0.5, -20, -0.2]
        
        # Extract trajectory data for jerk calculation
        ego_longitudial_positions = self.egovehicle.traj.reshape(-1, 2)[:, 0]
        
        # Calculate speed history (finite difference of position)
        ego_longitudial_speeds = (ego_longitudial_positions[1:] - ego_longitudial_positions[:-1]) / self.timestep if ego_longitudial_positions.shape[0] > 1 else [self.egovehicle.initial_speed]  
        
        # Calculate acceleration history (finite difference of speed)
        ego_longitudial_accs = (ego_longitudial_speeds[1:] - ego_longitudial_speeds[:-1]) / self.timestep if ego_longitudial_positions.shape[0] > 2 else [0]
        
        # Calculate jerk history (finite difference of acceleration)
        ego_longitudial_jerks = (ego_longitudial_accs[1:] - ego_longitudial_accs[:-1]) / self.timestep if ego_longitudial_positions.shape[0] > 3 else [0]
            
        # Mobility reward - normalized speed
        max_speed = 130/3.6  # Maximum speed in m/s (130 km/h for highway, 100 km/h for corner_case)
        ego_longitudial_speed = abs(ego_longitudial_speeds[-1])/max_speed
        
        # Comfort penalties
        ego_longitudial_acc = abs(ego_longitudial_accs[-1]) if ego_longitudial_positions.shape[0] > 2 else 0
        ego_longitudial_jerk = abs(ego_longitudial_jerks[-1]) if ego_longitudial_positions.shape[0] > 3 else 0
        
        # Only penalize excessive jerk
        if ego_longitudial_jerk <= 2:
            ego_longitudial_jerk = 0
            
        # Steering penalty
        ego_lanechange_angle = abs(self.egovehicle.angle) 

        # Safety penalties - front time gap (THWF)               
        FV_rel_distance = state_next[2]
        if FV_rel_distance > 40:
            ego_THWF = 0  # No penalty for far front vehicles
        else:
            if state_next[1] < 1:
                ego_THWF = 20  # High penalty for low speeds
            else: 
                # Time headway = distance / speed
                ego_THWF = (FV_rel_distance - self.Vissim.Net.Vehicles.ItemByKey(self.egovehicle.FVid).AttValue('Length')) / state_next[1]
            # Convert to exponential penalty (smaller gaps -> higher penalty)
            ego_THWF = math.exp(-ego_THWF)
            
        # Safety penalties - rear time gap (THWR)
        RV_rel_distance = abs(state_next[5])
        if RV_rel_distance > 40:
            ego_THWR = 0  # No penalty for far rear vehicles
        else:
            if self.Vissim.Net.Vehicles.ItemByKey(self.egovehicle.RVid).AttValue('Speed') / 3.6 < 1:
                ego_THWR = 20  # High penalty for low speeds
            else:
                # Time headway = distance / speed
                ego_THWR = (RV_rel_distance - self.egovehicle.length) / (self.Vissim.Net.Vehicles.ItemByKey(self.egovehicle.RVid).AttValue('Speed') / 3.6)
            # Convert to exponential penalty
            ego_THWR = math.exp(-ego_THWR)
        
        # Combined gap penalty
        ego_gap = ego_THWF + ego_THWR
                
        # Lateral position penalty (lane departure)
        lateral_beyond = 0
        if self.egovehicle.latpos > 2.75 * lanewidth:  # Too far right
            lateral_beyond = (self.egovehicle.latpos - 2.75 * lanewidth) / lanewidth
        elif self.egovehicle.latpos < 0.25 * lanewidth:  # Too far left
            lateral_beyond = (0.25 * lanewidth - self.egovehicle.latpos) / lanewidth
            
        # Collision detection
        collision = 0
        # Lane boundary collision
        if self.egovehicle.latpos >= 3.5 * lanewidth or self.egovehicle.latpos <= -0.5 * lanewidth:
            collision = 1
        
        # Front vehicle collision
        if abs(state_next[2]) < 15 and abs(state_next[2]) < self.Vissim.Net.Vehicles.ItemByKey(self.egovehicle.FVid).AttValue('Length'):
            collision = 1
        
        # Rear vehicle collision
        if abs(state_next[5]) < 15 and abs(state_next[5]) < self.egovehicle.length:
            collision = 1
        
        # Episode termination conditions
        if collision == 1 or self.Vissim.Simulation.AttValue('SimSec') > 194.9:  # Time limit (194.9s for highway, 7.9s for corner case)
            done = True
        
        # Combine reward components
        features = np.array([
            ego_longitudial_speed,         # Speed reward
            ego_longitudial_jerk,          # Jerk penalty
            ego_lanechange_angle,          # Steering angle penalty
            ego_gap,                       # Time gap penalty
            collision,                     # Collision penalty
            lateral_beyond                 # Lane departure penalty
        ])
        
        # Apply reward weights
        reward = np.dot(features, theta)
     
        return reward, done