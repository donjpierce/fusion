import math
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

"""
tritium mass = 3
deuterium mass = 2
neutron mass = 1
Helium mass = 4

The particle state is a list of dictionaries:
example_state = [{'type': 'tritium', 'id': 0, 'size': 0.3, 'mass': 3, 
          'position': [1, 0, 0], 'velocity': [0, 0,-1], 'acceleration': [1, 1, 0]},
           
         {'type': 'tritium', 'id': 1, 'size': 0.3, 'mass': 3, 
          'position': [1, 0, 1], 'velocity': [0, -1, -1], 'acceleration': [0, 0, 0]}, 
         
         {'type': 'deuterium', 'id': 2, 'size': 0.2, 'mass': 2, 
          'position': [0, 0.5, 1], 'velocity': [1, 0.2, 0.1], 'acceleration': [0, 0, 0]},
           
         {'type': 'neutron', 'id': 3, 'size': 0.1, 'mass': 1, 
          'position': [1, 1, 0], 'velocity': [5, 0, 0], 'acceleration': [0, 1, 1]}]
"""


class ParticleBox:
    def __init__(self, init_state, boundary_min=-2, boundary_max=2):
        """
        initializes the cube with particles, begins time, and sets the boundary of the simulation

        :param init_state: list:
                [{'type': string 'id': int, 'mass': scalar, 'position': [x, y, z], 'velocity': [vx, vy, vz]}]
        :param boundary: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        self.init_state = init_state
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.boundary_min = boundary_min
        self.boundary_max = boundary_max
        self.velocity_threshold = 10 * boundary_max  #
        self.r = 0.005  # inflection radius where nuclear force becomes greater than the EM forces
        self.massHe = 4
        self.massD = 2
        self.massT = 3
        self.massN = 1
        self.sizeHe = 0.004
        self.sizeD = 0.002
        self.sizeT = 0.002
        self.sizeN = 0.001

    def update(self, dt):
        """
        update frame once by dt seconds

        :param self: object
        :param dt: float
        """
        self.time_elapsed += dt

        # update positions
        # for i, particle in enumerate(self.state):
        #     particle['position'] += dt * np.array(particle['velocity'])  # forward Euler method
        # collect data about the temperature of the state
        particles = np.array([self.state[i] for i in np.arange(len(self.state))])
        temperature(particles, self.time_elapsed)

        """ 
        DESCRIPTION OF DISTANCE MATRIX:
            `distance' is a distance matrix between all the particles, where its indices correspond to particle ids
            e.g. if matrix element (1,2)=1.4 this means that particles with ids 1 and 2 are 1.4 units apart
            NOTE: this assumes only that the particle ids start from 0 and that the init_state dictionary is ordered
        """
        distance = squareform(pdist(np.array([self.state[i]['position'] for i in np.arange(len(self.state))])))

        inflection_r = self.r

        ind_short1, ind_short2= np.where(distance < inflection_r)  # particles interacting under the short-range force
        unique_short = (ind_short1 < ind_short2)
        ind_short1 = ind_short1[unique_short]
        ind_short2 = ind_short2[unique_short]

        ind_long1, ind_long2 = np.where((distance > inflection_r))
        # ind_long1, ind_long2 = np.where(distance < 100 * inflection_r)
        unique_long = (ind_long1 < ind_long2)
        ind_long1 = ind_long1[unique_long]
        ind_long2 = ind_long2[unique_long]

        for i1, i2 in zip(ind_short1, ind_short2):
            particle1_type = self.state[i1]['type']
            particle1_size = self.state[i1]['size']
            particle1_id = self.state[i1]['id']
            particle2_type = self.state[i2]['type']
            particle2_size = self.state[i2]['size']
            particle2_id = self.state[i2]['id']

            m1 = self.state[particle1_id]['mass']
            pos1 = np.array(self.state[particle1_id]['position'])
            vel1 = np.array(self.state[particle1_id]['velocity'])

            m2 = self.state[particle2_id]['mass']
            pos2 = np.array(self.state[particle2_id]['position'])
            vel2 = np.array(self.state[particle2_id]['velocity'])

            # determine relative position and velocity
            rel_pos1 = pos1 - pos2
            rel_pos2 = pos2 - pos1
            rel_vel1 = vel1 - vel2
            rel_vel2 = vel2 - vel1

            rel_pos1_squared = abs(np.dot(rel_pos1, rel_pos1))
            rel_pos2_squared = abs(np.dot(rel_pos2, rel_pos2))

            if rel_pos1_squared == 0:
                rel_pos1_squared = 0.001
            if rel_pos2_squared == 0:
                rel_pos2_squared = 0.001

            rel_pos1_unit = np.array([rel_pos1[i] / math.sqrt(rel_pos1_squared) for i in range(3)])
            rel_pos2_unit = np.array([rel_pos1[i] / math.sqrt(rel_pos2_squared) for i in range(3)])

            # the attractive force due to the Yukawa potential is applied
            acel_scalar1 = -(1 + 1 / math.sqrt(rel_pos1_squared)) * \
                            (math.exp(-m1 * math.sqrt(rel_pos1_squared)) / math.sqrt(rel_pos1_squared)) / m1
            acel_scalar2 = -(1 + 1 / math.sqrt(rel_pos2_squared)) * \
                            (math.exp(-m2 * math.sqrt(rel_pos2_squared)) / math.sqrt(rel_pos2_squared)) / m2

            acel1 = np.array([acel_scalar1 * rel_pos1_unit[i] for i in range(3)])  # forward Euler
            acel2 = np.array([acel_scalar2 * rel_pos2_unit[i] for i in range(3)])  # forward Euler

            # move the particle according to the timestep
            vel1 = np.array([vel1[i] + acel1[i] * dt for i in range(3)])  # forward Euler
            vel2 = np.array([vel2[i] + acel2[i] * dt for i in range(3)])  # forward Euler
            pos1 = np.array([pos1[i] + vel1[i] * dt for i in range(3)])  # forward Euler
            pos2 = np.array([pos2[i] + vel2[i] * dt for i in range(3)])  # forward Euler

            # update the states of the particles
            self.state[particle1_id]['position'] = np.array(pos1)
            self.state[particle1_id]['velocity'] = np.array(vel1)
            self.state[particle1_id]['acceleration'] = np.array(acel1)

            self.state[particle2_id]['position'] = np.array(pos2)
            self.state[particle2_id]['velocity'] = np.array(vel2)
            self.state[particle2_id]['acceleration'] = np.array(acel2)

            # momentum vector of the center of mass frame
            vel_cm = (m1 * vel1 + m2 * vel2) / (m1 + m2)

            # collisions of spheres reflect rel_vel over rel_pos
            three_momentum1 = np.dot(rel_vel1, rel_pos1)
            three_momentum2 = np.dot(rel_vel2, rel_pos2)
            rel_vel1 = 2 * rel_pos1 * three_momentum1 / rel_pos1_squared - rel_vel1
            rel_vel2 = 2 * rel_pos2 * three_momentum2 / rel_pos2_squared - rel_vel2

            if math.sqrt(rel_pos1_squared) <= particle1_size + particle2_size:
                if (particle1_type == 'tritium' and particle2_type == 'deuterium') or \
                        (particle1_type == 'deuterium' and particle2_type == 'tritium'):
                    record_fusion(1, self.time_elapsed)
                    print('fusion occured at: ' + str(self.time_elapsed))
                    # a D-T reaction involves producing a new Helium particle and a neutron
                    new_init_acceleration = np.array([0, 0, 0])
                    new_he_velocity = (vel1 + vel2) / self.massHe
                    new_he_particle = {'type': 'helium', 'id': particle1_id, 'size': self.sizeHe,
                                       'mass': self.massHe, 'position': pos1, 'velocity': new_he_velocity,
                                       'acceleration': new_init_acceleration}
                    new_neutron = {'type': 'neutron', 'id': particle2_id, 'size': self.sizeN,
                                   'mass': self.massN, 'position': pos2, 'velocity': random_vector(),
                                   'acceleration': new_init_acceleration}

                    # remove old deuterium - tritium pair from state
                    if particle1_id > particle2_id:
                        self.state.pop(particle1_id)
                        self.state.pop(particle2_id)
                    else:
                        self.state.pop(particle2_id)
                        self.state.pop(particle1_id)

                    # add new particles to state
                    self.state.append(new_he_particle)
                    self.state.append(new_neutron)

                    # correct the order of the new state: sort by particle id
                    self.state.sort(key=lambda d: d['id'])
                else:
                    # any other reaction involves a normal elastic collision between the particles
                    self.state[particle1_id]['velocity'] = vel_cm + rel_vel1 * m2 / (m1 + m2)
                    self.state[particle2_id]['velocity'] = vel_cm - rel_vel2 * m1 / (m1 + m2)

        for i1, i2 in zip(ind_long1, ind_long2):
            particle1_id = self.state[i1]['id']
            particle1_type = self.state[i1]['type']
            particle2_id = self.state[i2]['id']
            particle2_type = self.state[i2]['type']

            if (particle1_type == 'neutron') or (particle2_type == 'neutron'):
                continue  # neutrons do not experience the long-range electromagnetic forces

            m1 = self.state[particle1_id]['mass']
            pos1 = np.array(self.state[particle1_id]['position'])
            vel1 = np.array(self.state[particle1_id]['velocity'])

            m2 = self.state[particle2_id]['mass']
            pos2 = np.array(self.state[particle2_id]['position'])
            vel2 = np.array(self.state[particle2_id]['velocity'])

            # determine relative position
            rel_pos1 = pos1 - pos2
            rel_pos2 = pos2 - pos1

            rel_pos1_squared = abs(np.dot(rel_pos1, rel_pos1))
            rel_pos2_squared = abs(np.dot(rel_pos2, rel_pos2))

            rel_pos_unit1 = [rel_pos1[i] / math.sqrt(rel_pos1_squared) for i in range(3)]
            rel_pos_unit2 = [rel_pos2[i] / math.sqrt(rel_pos2_squared) for i in range(3)]

            # a simple 1/r^2 repulsive Coloumb force is applied
            acel_scalar1 = (1 / rel_pos1_squared) / m1
            acel_scalar2 = (1 / rel_pos2_squared) / m2
            acel1 = [acel_scalar1 * rel_pos_unit1[i] for i in range(3)]
            acel2 = [acel_scalar2 * rel_pos_unit2[i] for i in range(3)]

            # move the particle according to the timestep
            vel1 = [vel1[i] + acel1[i] * dt for i in range(3)]
            vel2 = [vel2[i] + acel2[i] * dt for i in range(3)]
            pos1 = [pos1[i] + vel1[i] * dt for i in range(3)]
            pos2 = [pos2[i] + vel2[i] * dt for i in range(3)]

            # update the states of the particles
            self.state[particle1_id]['position'] = pos1
            self.state[particle1_id]['velocity'] = vel1
            self.state[particle1_id]['acceleration'] = acel1

            self.state[particle2_id]['position'] = pos2
            self.state[particle2_id]['velocity'] = vel2
            self.state[particle2_id]['acceleration'] = acel2

        # check for crossing boundary
        for particle in self.state:
            for component in np.arange(len(particle['position'])):
                crossed_min = (particle['position'][component] < self.boundary_min + particle['size'])
                crossed_max = (particle['position'][component] > self.boundary_max - particle['size'])
                if crossed_min:
                    particle['position'][component] = self.boundary_min + particle['size']
                    particle['velocity'][component] *= -1
                if crossed_max:
                    particle['position'][component] = self.boundary_max - particle['size']
                    particle['velocity'][component] *= -1


def random_vector():
    return np.array([random.random() * random_sign(),
                     random.random() * random_sign(),
                     random.random() * random_sign()])


def random_sign():
    if random.random() < 0.5:
        return -1
    else:
        return 1


times = []
av_tritium_temps = []
av_deuterium_temps = []
av_helium_temps = []
av_neutron_temps = []


def temperature(particles, timestamp):
    """
    Function receives particle dict for every particle for each timestep dt
    :param particles: array: array of particle dictionaries in the state
    :param timestamp: int: time corresponding to particles
    :return: 0
    # """
    tritium_temp = []
    deuterium_temp = []
    helium_temp = []
    neutron_temp = []
    for particle in particles:
        if particle['type'] == 'tritium':
            tritium_temp.append(np.dot(particle['velocity'], particle['velocity']))
        else:
            tritium_temp.append(0)
        if particle['type'] == 'deuterium':
            deuterium_temp.append(np.dot(particle['velocity'], particle['velocity']))
        else:
            deuterium_temp.append(0)
        if particle['type'] == 'helium':
            helium_temp.append(np.dot(particle['velocity'], particle['velocity']))
        else:
            helium_temp.append(0)
        if particle['type'] == 'neutron':
            neutron_temp.append(np.dot(particle['velocity'], particle['velocity']))
        else:
            neutron_temp.append(0)
    times.append(timestamp)
    av_tritium_temps.append(sum(tritium_temp) / len(tritium_temp))
    av_deuterium_temps.append(sum(deuterium_temp) / len(deuterium_temp))
    av_helium_temps.append(sum(helium_temp) / len(helium_temp))
    av_neutron_temps.append(sum(neutron_temp) / len(neutron_temp))
    return 0


def plot_temperature(axis1, axis2):
    """
    plots the data
    :param axis1: array: times
    :param axis2: quadruple: (tritium_temp, deuterium_temp, helium_temp, neutron_temp)
    :return: 0
    """
    fig2, ax_temp = plt.subplots()
    ax_temp.plot(axis1, axis2[0], 'r-')
    ax_temp.plot(axis1, axis2[1], 'b-')
    ax_temp.plot(axis1, axis2[2], 'g-')
    ax_temp.plot(axis1, axis2[3], 'y-')
    ax_temp.set_xlabel('Time')
    ax_temp.set_ylabel('Average Velocity of Particles')
    ax_temp.legend(['Tritium', 'Deuterium', 'Helium', 'Neutron'])
    plt.savefig('AverageVelocitiesOverTime.png')
    return 0


# set up initial state
counts = []
count_times = []
rates = []


def record_fusion(count, timestep):
    counts.append(count)
    count_times.append(timestep)
    rates.append(sum(counts) / timestep)
    return 0


def plot_fusion_counts(axis1, axis2):
    fig3, ax_counts = plt.subplots()
    ax_counts.plot(axis1, axis2, 'g-')
    ax_counts.set_xlabel('Time')
    ax_counts.set_ylabel('Number of Fusion Reactions per Second')
    plt.savefig('RateOfFusion.png')
    return 0


np.random.seed(1)
plot_boundary = 2

# fill the init_state dictionary with N randomly-positioned particles moving at random velocity
N = 5
types = ['tritium', 'deuterium']
mass = {'tritium': 2, 'deuterium': 2}
size = {'tritium': 0.2, 'deuterium': 0.2}
N_types = {'tritium': 0, 'helium': 0, 'deuterium': 0, 'neutron': 0}

particles = np.random.choice(types, N)
init_state = []
for i, particle in enumerate(particles):
    init_state.append(
        {'type': particle, 'id': i, 'mass': mass[particle], 'size': size[particle],
         'position': random_vector() * plot_boundary, 'velocity': random_vector(), 'acceleration': [0, 0, 0]}
       )
    N_types[particle] += 1

box = ParticleBox(init_state)
dt = 1 / 100

# set up figure and animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
# ax.axis('off')

# initialize particle points
tritium = sum([ax.plot([], [], [], 'ro', ms=2) for i in np.arange(N)], [])
deuterium = sum([ax.plot([], [], [], 'bo', ms=2) for i in np.arange(N)], [])
helium = sum([ax.plot([], [], [], 'go', ms=4) for i in np.arange(N)], [])
neutron = sum([ax.plot([], [], [], 'yo', ms=1) for i in np.arange(N)], [])

# prepare the axis limits and labels
ax.set_xlim3d([-plot_boundary, plot_boundary])
ax.set_xlabel('x')
ax.set_ylim3d([-plot_boundary, plot_boundary])
ax.set_ylabel('y')
ax.set_zlim3d([-plot_boundary, plot_boundary])
ax.set_zlabel('z')

# set a point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# set title
ax.set_title('My Fusion Reactor')


def init():
    """
    initializes the animation
    :return: particles, cube
    """
    for trit, deut, hel, neut in zip(tritium, deuterium, helium, neutron):
        trit.set_data([], [])
        trit.set_3d_properties([])

        deut.set_data([], [])
        deut.set_3d_properties([])

        hel.set_data([], [])
        hel.set_3d_properties([])

        neut.set_data([], [])
        neut.set_3d_properties([])
    return tritium + deuterium + helium + neutron


def animate(i):
    """
    perform animation step
    :param i:
    :return:
    """
    global box, dt, ax, fig
    box.update(dt)
    # ms = int(fig.dpi * 2 * box.size * fig.get_figwidth() / np.diff(ax.get_xbound())[0])

    for trit, deut, hel, neut, particle in zip(tritium, deuterium, helium, neutron, box.state):
        x = particle['position'][0]
        y = particle['position'][1]
        z = particle['position'][2]
        if particle['type'] == 'tritium':
            trit.set_data(x, y)
            trit.set_3d_properties(z)
        if particle['type'] == 'deuterium':
            deut.set_data(x, y)
            deut.set_3d_properties(z)
        if particle['type'] == 'helium':
            hel.set_data(x, y)
            hel.set_3d_properties(z)
        if particle['type'] == 'neutron':
            neut.set_data(x, y)
            neut.set_3d_properties(z)

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return tritium + deuterium + helium + neutron


# create animation object
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=30, blit=True)
plt.show()
# ani.save('fusion.html', fps=30, extra_args=['-vcodec', 'libx264'])

# plot temperature
plot_temperature(times, (av_tritium_temps, av_deuterium_temps, av_helium_temps, av_neutron_temps))
# plot rate of fuson reactions
plot_fusion_counts(count_times, rates)
