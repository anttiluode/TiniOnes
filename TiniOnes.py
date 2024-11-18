import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk 
from tkinter import ttk, filedialog, messagebox
from math import cos, sin, radians, sqrt, atan2, degrees
import cv2
from typing import List, Tuple, Dict
import random
import threading
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame  # For audio playback

# Initialize pygame mixer for audio
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Configuration and Parameters
frequency_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 49),
    'low_gamma': (49, 60),
    'high_gamma': (60, 70)
}

latent_dim = 64
eeg_batch_size = 64
eeg_epochs = 50
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

set_seed(42)

class EEGAutoencoder(nn.Module):
    def __init__(self, channels=5, frequency_bands=7, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2))
        )
        
        # This matches the saved state dimensions
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(160, latent_dim)  # 160 = 32 * 5 * 1
        self.fc2 = nn.Linear(latent_dim, 160)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(1,2), stride=(1,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(1,2), stride=(1,2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.fc1(x)
        x = self.fc2(latent)
        x = x.view(-1, 32, 5, 1)  # Reshape to match decoder input
        x = self.decoder(x)
        x = x.squeeze(1)  # Remove channel dimension
        return x, latent

class BrainCoupler:
    def __init__(self, eeg_model, small_brain, coupling_rate=0.1):
        self.eeg_model = eeg_model
        self.small_brain = small_brain
        self.coupling_rate = coupling_rate
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(small_brain.parameters(), lr=0.001)
        
    def train_step(self, eeg_data, t):
        if len(eeg_data.shape) == 2:
            eeg_data = eeg_data.unsqueeze(0).unsqueeze(0)
            
        # Get EEG latent vector
        with torch.no_grad():
            _, eeg_latent = self.eeg_model(eeg_data)
        
        # Detach latent vector to prevent backward through EEG model
        eeg_latent = eeg_latent.detach()
        
        # Get small brain output
        brain_output = self.small_brain(eeg_latent, t)
        
        # Compute loss
        loss = self.loss_fn(brain_output, eeg_latent)
        
        # Backward pass with retain_graph
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return loss.item()

class SmallBrain(nn.Module):
    def __init__(self, num_neurons=16, latent_dim=64, coupling_strength=0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.latent_dim = latent_dim
        
        # Wave parameters
        self.frequencies = nn.Parameter(torch.rand(num_neurons) * 2.0)
        self.phases = nn.Parameter(torch.rand(num_neurons) * 2 * np.pi)
        self.amplitudes = nn.Parameter(torch.rand(num_neurons) * 0.5 + 0.5)
        
        # Coupling matrix
        self.coupling = nn.Parameter(torch.randn(num_neurons, num_neurons) * coupling_strength)
        
        # Neural projections
        self.input_proj = nn.Linear(latent_dim + 1, num_neurons)  # +1 for hearing neuron
        self.output_proj = nn.Linear(num_neurons, latent_dim)
        
        # Register state as a buffer to exclude it from gradient computations
        self.register_buffer('state', torch.zeros(num_neurons))
        self.memory = []
        
    def forward(self, x, t, hearing_input):
        # Handle device consistency
        if self.state.device != x.device:
            self.state = self.state.to(x.device)
        
        # Concatenate hearing input
        x = torch.cat([x, hearing_input.unsqueeze(0)], dim=1)
        
        # Project input to neuron space
        neuron_input = self.input_proj(x)
        
        # Generate oscillations
        t_tensor = torch.tensor(t, dtype=torch.float32, device=x.device)
        oscillations = self.amplitudes * torch.sin(
            2 * np.pi * self.frequencies * t_tensor + self.phases
        ).unsqueeze(0).repeat(x.size(0), 1)
        
        # Update state without tracking gradients
        with torch.no_grad():
            self.state += 0.1 * (
                oscillations[0] + 
                torch.matmul(self.state, self.coupling) +
                neuron_input[0]
            )
            self.state = torch.tanh(self.state)
        
        # Project to output space
        output = self.output_proj(self.state.unsqueeze(0))
        
        # Update memory
        self.memory.append(self.state.clone())
        if len(self.memory) > 100:
            self.memory.pop(0)
        
        return output

class EEGWaveNeuron:
    def __init__(self, frequency=None, amplitude=None, phase=None, memory_size=100):
        self.frequency = frequency if frequency is not None else np.random.uniform(0.1, 1.0)
        self.amplitude = amplitude if amplitude is not None else np.random.uniform(0.5, 1.0)
        self.phase = phase if phase is not None else np.random.uniform(0, 2 * np.pi)
        self.output = 0.0
        self.memory = np.zeros(memory_size)
        self.memory_pointer = 0
        self.resonance_weight = 0.9

    def activate(self, input_signal, eeg_signal, t):
        eeg_influence = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase) * eeg_signal
        past_resonance = np.mean(self.memory) * self.resonance_weight
        total_input = eeg_influence + input_signal + past_resonance

        # Apply tanh activation function
        self.output = np.tanh(total_input)

        # Check for NaN or Inf
        if not np.isfinite(self.output):
            self.output = 0.0

        # Update memory
        self.memory[self.memory_pointer] = self.output
        self.memory_pointer = (self.memory_pointer + 1) % len(self.memory)
        return self.output


class ResonantBrain:
    def __init__(self, num_neurons=16, memory_size=100):
        self.neurons = [EEGWaveNeuron(memory_size=memory_size) for _ in range(num_neurons)]
        self.connections = self._initialize_connections()

    def _initialize_connections(self):
        connections = {}
        for n1 in self.neurons:
            for n2 in self.neurons:
                if n1 != n2:
                    connections[(n1, n2)] = np.random.uniform(0.1, 0.5)
        return connections

    def update(self, eeg_latent: np.ndarray, dt=0.1):
        # Update neurons
        for neuron, latent_value in zip(self.neurons, eeg_latent):
            input_signal = np.mean([
                self.connections.get((neuron, other), 0) * other.output
                for other in self.neurons if other != neuron
            ])
            neuron.activate(input_signal, latent_value, dt)

        # Hebbian Learning
        for (pre, post), weight in self.connections.items():
            if pre.output > 0.5 and post.output > 0.5:
                self.connections[(pre, post)] = min(weight + 0.01, 1.0)
            else:
                self.connections[(pre, post)] = max(weight - 0.01, 0.0)

class DynamicWaveEEGProcessor:
    def __init__(self, eeg_model_path: str, latent_dim=64, num_neurons=16):
        self.eeg_model = EEGAutoencoder(channels=5, frequency_bands=7, latent_dim=latent_dim)
        try:
            self.eeg_model.load_state_dict(torch.load(eeg_model_path, map_location='cpu', weights_only=True), strict=False)
        except TypeError:
            self.eeg_model.load_state_dict(torch.load(eeg_model_path, map_location='cpu'))
        self.eeg_model.eval()
        self.brain = ResonantBrain(num_neurons=num_neurons)
        self.time = 0.0
        self.memory = []
        self.mouse_position = (0, 0)  # For audio zoom

    def process_and_update(self, eeg_data: np.ndarray) -> np.ndarray:
        # Reshape eeg_data to correct dimensions [batch, channels, frequency_bands]
        if len(eeg_data.shape) == 3:
            eeg_data = eeg_data.squeeze(0)  # Remove extra dimension if present
        if len(eeg_data.shape) == 2:
            eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected EEG data shape: {eeg_data.shape}")

        with torch.no_grad():
            _, latent_vector = self.eeg_model(eeg_tensor)
            
        latent_vector = latent_vector.squeeze(0).numpy()
        self.brain.update(latent_vector, dt=0.1)
        self.time += 0.1

        self.memory.append(latent_vector)
        if len(self.memory) > 50:
            self.memory.pop(0)

        return latent_vector

class TiniOne:
    def __init__(self, canvas_width: int, canvas_height: int, color: str, name: str, 
                 processor: DynamicWaveEEGProcessor, bug_radius: int = 20, num_waveneurons: int = 16):
        self.position = [random.randint(bug_radius, canvas_width - bug_radius),
                        random.randint(bug_radius, canvas_height - bug_radius)]
        self.direction = random.uniform(0, 360)
        self.speed = 5.0
        self.color = color
        self.name = name
        self.processor = processor
        self.bug_radius = bug_radius
        self.vision_angle = 90
        self.vision_range = 100
        self.vision_direction = self.direction  # Independent vision direction
        self.state = "exploring"
        self.trail = []
        self.can_talk = False
        self.genetic_traits = self.initialize_genetic_traits()
        self.echo_trails = []
        self.num_waveneurons = num_waveneurons
        self.particles = []  # For particle effects
        self.neuron_count = num_waveneurons  # Start with initial neuron count
        # Audio attributes
        self.sound_channel = pygame.mixer.Channel(random.randint(0, 7))  # Assign each TiniOne a different channel
        self.sound_frequency = None  # To store the current frequency
        self.last_sound_time = 0  # To control sound intervals
        self.sound_interval = random.uniform(4.0, 20.0)  # Longer intervals
        self.sound_memory = []  # For longer sound sequences
        self.heard_sounds = []  # For processing heard sounds
        self.communication_length = 0.8  # Default communication length
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def initialize_genetic_traits(self) -> dict:
        return {
            'trail_thickness': random.uniform(1.0, 3.0),
            'echo_duration': random.randint(5, 15),
            'echo_spread': random.uniform(0.5, 1.5),
            'draw_activation_threshold': random.uniform(0.6, 0.9)
        }

    def can_move_to(self, new_position, other_tiniones):
        for other in other_tiniones:
            if other == self:
                continue
            dist = sqrt((new_position[0] - other.position[0]) ** 2 +
                        (new_position[1] - other.position[1]) ** 2)
            if dist < self.bug_radius * 2:
                return False
        # Check boundaries
        if not (self.bug_radius <= new_position[0] <= self.canvas_width - self.bug_radius and
                self.bug_radius <= new_position[1] <= self.canvas_height - self.bug_radius):
            return False
        return True

    def move(self, other_tiniones):
        dx = cos(radians(self.direction)) * self.speed
        dy = sin(radians(self.direction)) * self.speed
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        new_position = [new_x, new_y]
        if self.can_move_to(new_position, other_tiniones):
            self.position = new_position
            self.trail.append(tuple(self.position))
            if len(self.trail) > 50:
                self.trail.pop(0)
        else:
            # Randomly adjust direction to find a valid movement
            self.direction += random.uniform(-45, 45)
            self.direction %= 360

    def detect_in_vision(self, other_tiniones: List['TiniOne'], webcam_input: np.ndarray) -> np.ndarray:
        vision_data = []
        for other in other_tiniones:
            if other == self:
                continue
            dx = other.position[0] - self.position[0]
            dy = other.position[1] - self.position[1]
            distance = sqrt(dx**2 + dy**2)
            if distance > self.vision_range:
                continue
            angle_to_other = degrees(atan2(dy, dx)) % 360
            angle_diff = (angle_to_other - self.vision_direction) % 360
            if angle_diff > 180:
                angle_diff -= 360
            if abs(angle_diff) <= self.vision_angle / 2:
                vision_data.append((distance, angle_to_other))

        vision_signal = np.zeros((5, 7))
        if webcam_input is not None:
            normalized_brightness = np.mean(webcam_input) / 255.0
            vision_signal[0, 0] = normalized_brightness

        for i, (distance, angle) in enumerate(vision_data):
            normalized_dist = max(0, (self.vision_range - distance) / self.vision_range)
            vision_signal[i % 5, i % 7] += normalized_dist
        return vision_signal

    def create_echo_trace(self, x: float, y: float):
        echo = {
            'position': (x, y),
            'thickness': self.genetic_traits['trail_thickness'],
            'duration': self.genetic_traits['echo_duration'],
            'spread': self.genetic_traits['echo_spread'],
            'remaining': self.genetic_traits['echo_duration']
        }
        self.echo_trails.append(echo)

    def create_particles(self):
        for _ in range(10):
            particle = {
                'position': self.position.copy(),
                'velocity': [random.uniform(-2, 2), random.uniform(-2, 2)],
                'life': random.randint(5, 15)
            }
            self.particles.append(particle)

    def update_particles(self):
        for particle in self.particles.copy():
            particle['position'][0] += particle['velocity'][0]
            particle['position'][1] += particle['velocity'][1]
            particle['life'] -= 1
            if particle['life'] <= 0:
                self.particles.remove(particle)

    def is_near(self, other_tinion, threshold=200):
        dx = self.position[0] - other_tinion.position[0]
        dy = self.position[1] - other_tinion.position[1]
        distance = sqrt(dx**2 + dy**2)
        return distance < threshold

    def hear_sound(self, frequency, volume):
        # Volume is adjusted based on distance
        self.heard_sounds.append((frequency, volume))

    def think_and_act(self, environment_input: np.ndarray, other_tiniones: List['TiniOne'], 
                     webcam_input: np.ndarray, neuron_boosts: List['NeuronBoost'], is_paused: bool,
                     volume: float, communication_length: float):
        if is_paused:
            return None, "", self.echo_trails

        vision_signal = self.detect_in_vision(other_tiniones, webcam_input)
        combined_input = environment_input + vision_signal
        hearing_input = torch.tensor([np.mean([v for f, v in self.heard_sounds])] if self.heard_sounds else [0.0], dtype=torch.float32)
        latent_vector = self.processor.process_and_update(combined_input)
        oscillatory_energy = np.mean([abs(neuron.output) for neuron in self.processor.brain.neurons])

        # Adjust speed based on EEG data
        eeg_activity = np.mean(combined_input)
        self.speed = 2.0 + eeg_activity * 5.0  # Speed ranges from 2 to 7

        # Adjust vision range and angle
        self.adjust_vision()

        if self.state == "exploring":
            self.direction += random.uniform(-15, 15) * oscillatory_energy
            self.vision_direction += random.uniform(-10, 10)  # Randomly adjust vision direction
        elif self.state == "avoiding":
            self.direction += random.uniform(-30, 30) * oscillatory_energy
            self.vision_direction += random.uniform(-20, 20)  # Adjust vision direction more drastically

        self.direction %= 360
        self.vision_direction %= 360
        self.move(other_tiniones)
        self.update_particles()

        # Check for neuron boosts
        for boost in neuron_boosts:
            if boost.is_collected(self.position, self.bug_radius):
                neuron_boosts.remove(boost)
                self.neuron_count += boost.boost_amount
                # Regenerate the brain with more neurons
                self.processor.brain = ResonantBrain(num_neurons=self.neuron_count)
                break  # Only collect one boost at a time

        # Store outputs for sound memory
        combined_output = latent_vector
        mean_output = np.mean(combined_output)
        self.sound_memory.append(mean_output)
        if len(self.sound_memory) > 20:
            self.sound_memory.pop(0)

        # Generate sound
        current_time = time.time()
        if current_time - self.last_sound_time > self.sound_interval:
            self.generate_sound(volume=volume, communication_length=communication_length)
            self.last_sound_time = current_time
            self.sound_interval = random.uniform(4.0, 20.0)  # Play sounds less frequently

        # Process heard sounds
        if self.heard_sounds:
            # Adjust speed based on heard sounds
            total_volume = sum([v for f, v in self.heard_sounds])
            if total_volume > 0:
                weighted_frequency = sum([f * v for f, v in self.heard_sounds]) / total_volume
                self.speed += (weighted_frequency - 1250) / 500  # Adjust speed slightly
                self.speed = max(2.0, min(7.0, self.speed))
            # Clear heard_sounds
            self.heard_sounds = []

        return combined_output, "", self.echo_trails

    def adjust_vision(self):
        # Logic to adjust vision range and angle inversely
        # For simplicity, let's oscillate vision range between 50 and 200
        self.vision_range += random.uniform(-5, 5)
        self.vision_range = max(50, min(200, self.vision_range))
        # Vision angle inversely proportional to vision range
        min_angle, max_angle = 30, 120
        self.vision_angle = max_angle - ((self.vision_range - 50) / (200 - 50)) * (max_angle - min_angle)
        self.vision_angle = max(min_angle, min(max_angle, self.vision_angle))

    def generate_sound(self, volume=1.0, communication_length=0.8):
        if not self.sound_memory:
            return

        duration = communication_length  # Use communication length set by user
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        num_samples = len(t)

        # Interpolate sound_memory to match num_samples
        sound_memory_array = np.array(self.sound_memory)
        sound_memory_interpolated = np.interp(
            np.linspace(0, len(sound_memory_array), num_samples),
            np.arange(len(sound_memory_array)),
            sound_memory_array
        )

        # Map sound_memory_array values to frequencies
        frequencies = 1000 + ((sound_memory_interpolated + 1) / 2) * 500  # Map from [-1,1] to [1000,1500] Hz

        # Generate waveform with varying frequency
        waveform = np.sin(2 * np.pi * frequencies * t)

        # Adjust volume based on proximity to mouse (Audio Zoom)
        mouse_x, mouse_y = self.processor.mouse_position
        dx = self.position[0] - mouse_x
        dy = self.position[1] - mouse_y
        distance = sqrt(dx**2 + dy**2)
        max_distance = 200  # Maximum distance for audio zoom
        volume_factor = max(0.1, 1 - (distance / max_distance))
        volume *= volume_factor

        # Create sound
        sound_array = np.stack([waveform * volume, waveform * volume], axis=-1)  # Stereo sound
        sound = pygame.sndarray.make_sound((sound_array * 32767).astype(np.int16))

        # Play the sound
        self.sound_channel.play(sound)

        # TiniOne hears its own sound slightly
        self.hear_sound(np.mean(frequencies), volume * 0.5)  # They hear themselves at half volume

    def draw(self, canvas):
        x, y = self.position
        # Draw the minion-like TiniOne
        bug_image = self.create_minion_image()
        bug_photo = ImageTk.PhotoImage(bug_image)
        canvas.create_image(x, y, image=bug_photo)
        # Keep a reference to prevent garbage collection
        self.bug_photo = bug_photo

        # Draw vision cone
        self.draw_vision_cone(canvas)

    def create_minion_image(self):
        size = self.bug_radius * 2
        image = Image.new('RGBA', (size, size))
        draw = ImageDraw.Draw(image)

        # Yellow body with gradient edges
        for i in range(size // 2):
            color = (255, 255, int(255 * (i / (size / 2))), int(255 * (1 - i / (size / 2))))
            draw.ellipse(
                [i, i, size - i, size - i],
                fill=color
            )

        # Green overalls
        draw.polygon(
            [(0, size), (size, size), (size, size * 0.6), (0, size * 0.6)],
            fill=(0, 128, 0)
        )

        # Simple eyes
        eye_radius = size * 0.1
        eye_x = size * 0.35
        eye_y = size * 0.3
        draw.ellipse(
            [eye_x - eye_radius, eye_y - eye_radius, eye_x + eye_radius, eye_y + eye_radius],
            fill='white'
        )
        draw.ellipse(
            [size - eye_x - eye_radius, eye_y - eye_radius, size - eye_x + eye_radius, eye_y + eye_radius],
            fill='white'
        )
        # Pupils
        pupil_radius = eye_radius * 0.5
        draw.ellipse(
            [eye_x - pupil_radius, eye_y - pupil_radius, eye_x + pupil_radius, eye_y + pupil_radius],
            fill='black'
        )
        draw.ellipse(
            [size - eye_x - pupil_radius, eye_y - pupil_radius, size - eye_x + pupil_radius, eye_y + pupil_radius],
            fill='black'
        )

        # Mop of hair
        hair_start = size * 0.2
        hair_end = size * 0.8
        for i in range(5):
            draw.line(
                [(size / 2, hair_start), (hair_start + i * (hair_end - hair_start) / 4, hair_start - size * 0.1)],
                fill='black', width=1
            )

        return image

    def draw_vision_cone(self, canvas):
        x, y = self.position
        vision_start = self.vision_direction - self.vision_angle / 2
        vision_end = self.vision_direction + self.vision_angle / 2

        end1 = (
            x + cos(radians(vision_start)) * self.vision_range,
            y + sin(radians(vision_start)) * self.vision_range
        )
        end2 = (
            x + cos(radians(vision_end)) * self.vision_range,
            y + sin(radians(vision_end)) * self.vision_range
        )

        canvas.create_polygon(
            [x, y, end1[0], end1[1], end2[0], end2[1]],
            fill=self.color, stipple="gray25", outline=""
        )

class EnhancedTiniOne(TiniOne):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.small_brain = SmallBrain(
            num_neurons=self.num_waveneurons,
            latent_dim=64
        )
        self.brain_coupler = BrainCoupler(
            self.processor.eeg_model,
            self.small_brain
        )
        self.learning_rate = 0.1
        
    def think_and_act(self, environment_input, other_tiniones, webcam_input, neuron_boosts, is_paused,
                      volume, communication_length):
        if is_paused:
            return None, "", self.echo_trails

        vision_signal = self.detect_in_vision(other_tiniones, webcam_input)
        combined_input = environment_input + vision_signal
        hearing_input = torch.tensor([np.mean([v for f, v in self.heard_sounds])] if self.heard_sounds else [0.0], dtype=torch.float32)
        latent_vector = self.processor.process_and_update(combined_input)
        oscillatory_energy = np.mean([abs(neuron.output) for neuron in self.processor.brain.neurons])

        # Adjust speed based on EEG data
        eeg_activity = np.mean(combined_input)
        self.speed = 2.0 + eeg_activity * 5.0  # Speed ranges from 2 to 7

        # Adjust vision range and angle
        self.adjust_vision()

        if self.state == "exploring":
            self.direction += random.uniform(-15, 15) * oscillatory_energy
            self.vision_direction += random.uniform(-10, 10)  # Randomly adjust vision direction
        elif self.state == "avoiding":
            self.direction += random.uniform(-30, 30) * oscillatory_energy
            self.vision_direction += random.uniform(-20, 20)  # Adjust vision direction more drastically

        self.direction %= 360
        self.vision_direction %= 360
        self.move(other_tiniones)
        self.update_particles()

        t = time.time()
        loss = self.brain_coupler.train_step(
            torch.FloatTensor(environment_input).unsqueeze(0),
            t
        )

        brain_output = self.small_brain(
            torch.FloatTensor(latent_vector),
            t,
            hearing_input  # Pass hearing input
        )

        combined_output = (
            latent_vector * (1 - self.learning_rate) + 
            brain_output.detach().numpy() * self.learning_rate
        )

        # Check for neuron boosts
        for boost in neuron_boosts:
            if boost.is_collected(self.position, self.bug_radius):
                neuron_boosts.remove(boost)
                self.neuron_count += boost.boost_amount
                # Regenerate the brain with more neurons
                self.processor.brain = ResonantBrain(num_neurons=self.neuron_count)
                break  # Only collect one boost at a time

        # Store outputs for sound memory
        mean_output = np.mean(combined_output)
        self.sound_memory.append(mean_output)
        if len(self.sound_memory) > 20:
            self.sound_memory.pop(0)

        # Generate sound
        current_time = time.time()
        if current_time - self.last_sound_time > self.sound_interval:
            self.generate_sound(volume=volume, communication_length=communication_length)
            self.last_sound_time = current_time
            self.sound_interval = random.uniform(4.0, 20.0)  # Play sounds less frequently

        # Process heard sounds
        if self.heard_sounds:
            # Adjust speed based on heard sounds
            total_volume = sum([v for f, v in self.heard_sounds])
            if total_volume > 0:
                weighted_frequency = sum([f * v for f, v in self.heard_sounds]) / total_volume
                self.speed += (weighted_frequency - 1250) / 500  # Adjust speed slightly
                self.speed = max(2.0, min(7.0, self.speed))
            # Clear heard_sounds
            self.heard_sounds = []

        return combined_output, "", self.echo_trails

class NeuronBoost:
    def __init__(self, position):
        self.position = position
        self.boost_amount = random.randint(1, 5)

    def draw(self, canvas):
        x, y = self.position
        canvas.create_text(x, y, text="+Neuron", fill="cyan", font=("Helvetica", 10))

    def is_collected(self, bug_position, bug_radius):
        dx = self.position[0] - bug_position[0]
        dy = self.position[1] - bug_position[1]
        distance = sqrt(dx**2 + dy**2)
        return distance < bug_radius

class EEGBugSimulatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EEG TiniOne Simulator")
        self.root.geometry("1200x800")
        self.root.resizable(False, False)

        self.model_path = tk.StringVar()
        self.webcam_index = tk.IntVar(value=0)
        self.background_image_path = tk.StringVar()
        self.num_waveneurons = tk.IntVar(value=16)
        self.simulation_running = False
        self.is_paused = False  # For pause/play control

        self.bug_speed = tk.DoubleVar(value=5.0)
        self.coupling_strength = tk.DoubleVar(value=0.1)
        self.num_tiniones = tk.IntVar(value=5)  # Default number of TiniOnes
        self.volume_level = tk.DoubleVar(value=1.0)  # Volume control
        self.communication_length = tk.DoubleVar(value=0.8)  # Communication length control

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.create_configuration_tab()
        self.create_simulation_tab()
        self.create_help_tab()

    def create_configuration_tab(self):
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text='Configuration')

        # Model Selection
        model_frame = ttk.LabelFrame(self.config_frame, text="1. Select EEG Autoencoder Model (.pth)", padding=10)
        model_frame.pack(fill=tk.X, padx=20, pady=10)

        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=80, state='readonly')
        self.model_entry.pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(model_frame, text="Browse", command=self.browse_model).pack(side=tk.LEFT)

        # Input Source Selection
        input_frame = ttk.LabelFrame(self.config_frame, text="2. Select Input Source", padding=10)
        input_frame.pack(fill=tk.X, padx=20, pady=10)

        self.input_option = tk.IntVar(value=1)
        ttk.Radiobutton(input_frame, text="Use Webcam", variable=self.input_option, value=1, 
                       command=self.toggle_input_option).grid(row=0, column=0, sticky='w', pady=5)
        ttk.Radiobutton(input_frame, text="Use Background Image", variable=self.input_option, value=2,
                       command=self.toggle_input_option).grid(row=1, column=0, sticky='w', pady=5)

        self.webcam_frame = ttk.Frame(input_frame)
        self.webcam_frame.grid(row=0, column=1, sticky='w', pady=5, padx=10)
        ttk.Label(self.webcam_frame, text="Webcam Index:").pack(side=tk.LEFT)
        self.webcam_spinbox = ttk.Spinbox(self.webcam_frame, from_=0, to=10, width=5, textvariable=self.webcam_index)
        self.webcam_spinbox.pack(side=tk.LEFT, padx=(5,0))

        self.image_frame = ttk.Frame(input_frame)
        self.image_frame.grid(row=1, column=1, sticky='w', pady=5, padx=10)
        self.image_entry = ttk.Entry(self.image_frame, textvariable=self.background_image_path, width=60, state='disabled')
        self.image_entry.pack(side=tk.LEFT, padx=(0,10))
        self.image_browse_button = ttk.Button(self.image_frame, text="Browse", command=self.browse_background_image, state='disabled')
        self.image_browse_button.pack(side=tk.LEFT)

        # TiniOne Configuration
        tinione_frame = ttk.LabelFrame(self.config_frame, text="3. Configure TiniOnes", padding=10)
        tinione_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(tinione_frame, text="Number of TiniOnes:").grid(row=0, column=0, sticky='w', pady=5)
        self.tinione_spinbox = ttk.Spinbox(tinione_frame, from_=1, to=20, increment=1, textvariable=self.num_tiniones, width=5)
        self.tinione_spinbox.grid(row=0, column=1, sticky='w', pady=5, padx=(5,0))

        ttk.Label(tinione_frame, text="Number of Wave Neurons per TiniOne:").grid(row=1, column=0, sticky='w', pady=5)
        self.neurons_spinbox = ttk.Spinbox(tinione_frame, from_=1, to=1500, increment=1, textvariable=self.num_waveneurons, width=5)
        self.neurons_spinbox.grid(row=1, column=1, sticky='w', pady=5, padx=(5,0))

        self.start_button = ttk.Button(self.config_frame, text="Start Simulation", command=self.start_simulation, state='disabled')
        self.start_button.pack(pady=20)

    def create_simulation_tab(self):
        self.simulation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.simulation_frame, text='Simulation')

        self.main_frame = tk.Frame(self.simulation_frame)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<Motion>", self.update_mouse_position)
        self.canvas.bind("<Double-Button-1>", self.drop_neuron_boost)  # Changed to double-click
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.sidebar = tk.Frame(self.main_frame, width=300, bg="grey")
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.cap = None
        self.background_image = None
        self.tiniones = []
        self.neuron_boosts = []
        self.simulation_running = False
        self.mouse_position = (0, 0)
        self.dragging_tinione = None  # For dragging TiniOnes

        # Create control panel
        self.create_control_panel()
        # Create heatmap display
        self.create_heatmap_display()

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.sidebar, text="Control Panel", padding=10)
        control_frame.pack(pady=10, fill=tk.X)

        ttk.Label(control_frame, text="TiniOne Speed").pack()
        self.speed_scale = ttk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.bug_speed)
        self.speed_scale.pack(fill=tk.X)

        ttk.Label(control_frame, text="Coupling Strength").pack()
        self.coupling_scale = ttk.Scale(control_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.coupling_strength)
        self.coupling_scale.pack(fill=tk.X)

        # Volume Control
        ttk.Label(control_frame, text="Volume Level").pack()
        self.volume_scale = ttk.Scale(control_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.volume_level)
        self.volume_scale.pack(fill=tk.X)

        # Communication Length Control
        ttk.Label(control_frame, text="Communication Length").pack()
        self.comm_length_scale = ttk.Scale(control_frame, from_=0.2, to=5.0, orient=tk.HORIZONTAL, variable=self.communication_length)
        self.comm_length_scale.pack(fill=tk.X)

        # Pause/Play Button
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(pady=5)

    def create_heatmap_display(self):
        self.heatmap_frame = tk.Frame(self.sidebar)
        self.heatmap_frame.pack(fill=tk.BOTH, expand=True)
        self.heatmap_canvases = {}
        self.heatmap_figures = {}

    def update_mouse_position(self, event):
        self.mouse_position = (event.x, event.y)

    def on_mouse_down(self, event):
        x, y = event.x, event.y
        for tinione in self.tiniones:
            dx = x - tinione.position[0]
            dy = y - tinione.position[1]
            distance = sqrt(dx**2 + dy**2)
            if distance <= tinione.bug_radius:
                self.dragging_tinione = tinione
                break

    def on_mouse_move(self, event):
        if self.dragging_tinione is not None:
            self.dragging_tinione.position[0] = event.x
            self.dragging_tinione.position[1] = event.y

    def on_mouse_up(self, event):
        self.dragging_tinione = None

    def drop_neuron_boost(self, event):
        position = (event.x, event.y)
        boost = NeuronBoost(position)
        self.neuron_boosts.append(boost)

    def toggle_input_option(self):
        option = self.input_option.get()
        if option == 1:
            self.webcam_spinbox.config(state='normal')
            self.background_image_path.set('')
            self.image_entry.config(state='disabled')
            self.image_browse_button.config(state='disabled')
        elif option == 2:
            self.webcam_spinbox.config(state='disabled')
            self.image_entry.config(state='normal')
            self.image_browse_button.config(state='normal')

    def browse_model(self):
        file_path = filedialog.askopenfilename(title="Select EEG Autoencoder Model", filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            self.model_path.set(file_path)
            self.check_ready_to_start()

    def browse_background_image(self):
        file_path = filedialog.askopenfilename(title="Select Background Image", 
                                             filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.background_image_path.set(file_path)

    def check_ready_to_start(self):
        if self.model_path.get():
            self.start_button.config(state='normal')
        else:
            self.start_button.config(state='disabled')

    def draw_particles(self, tinione: TiniOne):
        for particle in tinione.particles:
            x, y = particle['position']
            life_ratio = particle['life'] / 15
            color = f"#{int(255*life_ratio):02x}{int(255*life_ratio):02x}00"
            self.canvas.create_oval(
                x - 2, y - 2, x + 2, y + 2,
                fill=color, outline=""
            )

    def start_simulation(self):
        if self.simulation_running:
            messagebox.showwarning("Simulation Running", "The simulation is already running.")
            return

        if not self.model_path.get():
            messagebox.showwarning("No Model Selected", "Please select an EEG autoencoder model before starting.")
            return

        # Initialize input source
        if self.input_option.get() == 1:
            webcam_idx = self.webcam_index.get()
            self.cap = cv2.VideoCapture(webcam_idx)
            if not self.cap.isOpened():
                messagebox.showerror("Webcam Error", f"Cannot open webcam with index {webcam_idx}.")
                return
        else:
            bg_path = self.background_image_path.get()
            if not os.path.exists(bg_path):
                messagebox.showerror("Image Not Found", f"Background image not found at {bg_path}.")
                return
            self.background_image = Image.open(bg_path).resize((self.canvas_width, self.canvas_height))
            self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Initialize processor and TiniOnes
        processor = DynamicWaveEEGProcessor(
            eeg_model_path=self.model_path.get(),
            latent_dim=latent_dim,
            num_neurons=self.num_waveneurons.get()
        )
        processor.mouse_position = self.mouse_position  # For audio zoom

        self.tiniones = []
        for i in range(self.num_tiniones.get()):
            name = f"TiniOne_{i+1}"
            color = f"#{random.randint(0, 0xFFFFFF):06x}"
            tinione = EnhancedTiniOne(
                canvas_width=self.canvas_width,
                canvas_height=self.canvas_height,
                color=color,
                name=name,
                processor=processor,
                bug_radius=20,
                num_waveneurons=self.num_waveneurons.get()
            )
            self.tiniones.append(tinione)

        self.simulation_running = True
        self.create_bug_heatmaps()
        self.run_simulation()

    def create_bug_heatmaps(self):
        for tinione in self.tiniones:
            fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
            canvas = FigureCanvasTkAgg(fig, master=self.heatmap_frame)
            canvas.get_tk_widget().pack()
            self.heatmap_canvases[tinione.name] = canvas
            self.heatmap_figures[tinione.name] = (fig, ax)

    def update_bug_heatmaps(self):
        for tinione in self.tiniones:
            fig, ax = self.heatmap_figures[tinione.name]
            ax.clear()
            neuron_states = np.array([neuron.output for neuron in tinione.processor.brain.neurons])
            size = int(np.ceil(np.sqrt(len(neuron_states))))
            data = np.zeros((size, size))
            data.flat[:len(neuron_states)] = neuron_states
            ax.imshow(data, cmap='viridis', vmin=-1, vmax=1)
            ax.set_title(f"{tinione.name} ({tinione.neuron_count} neurons)", fontsize=8)
            ax.axis('off')
            self.heatmap_canvases[tinione.name].draw()

    def run_simulation(self):
        if not self.simulation_running:
            return

        self.canvas.delete("all")

        # Handle background
        if self.background_image is not None:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo)

        # Handle webcam
        webcam_input = None
        if self.input_option.get() == 1 and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (self.canvas_width, self.canvas_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                self.webcam_image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.webcam_image)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                webcam_input = cv2.resize(gray_frame, (self.canvas_width, self.canvas_height))

        # Environment input
        environment_input = np.random.rand(5, 7)

        # Draw neuron boosts
        for boost in self.neuron_boosts:
            boost.draw(self.canvas)

        # Update and draw TiniOnes
        for tinione in self.tiniones:
            latent_vector, _, echo_trails = tinione.think_and_act(
                environment_input, self.tiniones, webcam_input, self.neuron_boosts, self.is_paused,
                volume=self.volume_level.get(), communication_length=self.communication_length.get()
            )

            # Update processor's mouse position for audio zoom
            tinione.processor.mouse_position = self.mouse_position

            # Draw TiniOne
            tinione.draw(self.canvas)

            self.draw_particles(tinione)

            for echo in echo_trails.copy():
                if echo['remaining'] > 0:
                    self.draw_echo(echo)
                    echo['remaining'] -= 1
                else:
                    tinione.echo_trails.remove(echo)

        # Simulate hearing sounds from nearby TiniOnes
        for tinione in self.tiniones:
            for other_tinione in self.tiniones:
                if tinione != other_tinione:
                    if other_tinione.sound_frequency is not None:
                        dx = tinione.position[0] - other_tinione.position[0]
                        dy = tinione.position[1] - other_tinione.position[1]
                        distance = sqrt(dx**2 + dy**2)
                        max_hearing_distance = 200
                        if distance < max_hearing_distance:
                            volume = max(0.1, 1 - (distance / max_hearing_distance))
                            tinione.hear_sound(other_tinione.sound_frequency, volume)

        self.update_bug_heatmaps()

        self.root.after(50, self.run_simulation)

    def draw_echo(self, echo: dict):
        x, y = echo['position']
        thickness = echo['thickness']
        spread = echo['spread']
        self.canvas.create_oval(
            x - spread * 10, y - spread * 10,
            x + spread * 10, y + spread * 10,
            outline='white', width=thickness, fill=''
        )

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_button.config(text="Resume" if self.is_paused else "Pause")

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.simulation_running = False
        self.root.destroy()
        pygame.quit()

    def create_help_tab(self):
        self.help_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.help_frame, text='Help')

        help_text = """
        EEG TiniOne Simulator

        This simulator combines EEG data processing with neural oscillators to create interactive agents called TiniOnes.

        Key Features:
        - EEG Model Integration: Processes brain activity patterns.
        - SmallBrain System: Mini neural networks that learn from EEG patterns.
        - **Distance-Based Hearing**:
          - TiniOnes hear each other's sounds louder the closer they are to each other.
          - They have a neuron dedicated to processing heard sounds.
          - TiniOnes also hear themselves slightly.
        - **No Repulsion Behavior**:
          - TiniOnes do not change direction when they get close to each other.
          - Instead, they cannot overlap and adjust their movement to prevent overlapping.
        - **Adjustable Vision Cone**:
          - TiniOnes can adjust their vision range and angle.
          - When they look further, their vision cone becomes narrower.
          - They can adjust the direction of their vision independently of their movement.
        - **Pause and Play Controls**: Pause and resume the simulation at any time.
        - **Volume Level Control**: Adjust the overall volume of the TiniOnes' sounds.
        - **Communication Length Control**: Set the length of TiniOnes' communication (sound duration).
        - **Audio Zoom**: Sounds from TiniOnes become louder as you move the mouse closer to them.
        - Neuron Boosts: Users can drop boosts to increase TiniOnes' neurons.
          - **Double-Click** on the canvas to drop a neuron boost.
        - Click and Drag: Users can move TiniOnes by clicking and dragging them.
        - Vision Cones: Visual representation of TiniOnes' adjustable vision cones.
        - Minion-like Appearance: TiniOnes resemble minions with cute features.
        - Dynamic Visualization: See neural activity and interactions.
        - Neural Activity Heatmaps: View TiniOnes' neural heatmaps in the sidebar.
        - Particle Effects: Visualize TiniOne interactions.
        - Audio Feedback:
          - TiniOnes produce longer sound sequences based on their memory.
          - Sounds are played less frequently but have longer duration.
          - TiniOnes hear their own sounds slightly, influencing their behavior.
          - **Audio Zoom**: TiniOnes near the mouse cursor are louder.
        - Control Panel:
          - Adjust simulation parameters in real-time.
          - Pause and resume the simulation.
          - Control the volume level.
          - Set the communication length.

        **Understanding the TiniOnes' Brain:**
        TiniOnes have a brain modeled as a resonant neural network. They process inputs from their environment and other TiniOnes using wave-like neurons that generate oscillatory patterns. Their brains adapt over time through Hebbian learning, strengthening connections between neurons that activate together.

        - **EEG Processing**: TiniOnes use an EEG Autoencoder to extract latent features from the environment.
        - **SmallBrain Coupling**: A SmallBrain network learns to predict and respond to these EEG features.
        - **Resonant Brain**: The neurons in the TiniOnes' brains resonate and interact, creating complex behaviors.
        - **Memory**: TiniOnes have a memory of past neural states, influencing their current behavior and sounds.
        - **Sound Generation**: Sounds are generated based on their neural activity and memory, leading to unique audio patterns.
        - **Adjustable Vision**: TiniOnes can focus on distant objects with a narrower vision cone or nearby objects with a wider vision cone.
        - **Distance-Based Hearing**: TiniOnes hear sounds from others based on proximity, with closer sounds being louder.

        Usage:
        1. Select trained EEG model (.pth file).
        2. Choose input source (webcam/image).
        3. Configure number of TiniOnes and wave neurons.
        4. Start simulation.
        5. **Click and drag** TiniOnes to move them.
        6. **Double-click** on the canvas to drop neuron boosts.
        7. Use the **Pause** button to pause/resume the simulation.
        8. Adjust the **Volume Level** slider to control the sound volume.
        9. Adjust the **Communication Length** slider to set the length of TiniOnes' sounds.

        Behavior:
        - TiniOnes do not repel each other but cannot overlap. They adjust their movement to avoid occupying the same space.
        - They hear each other's sounds louder when closer, influencing their neural activity and behavior.
        - TiniOnes also hear themselves slightly, adding to their sensory input.

        Enjoy exploring the emergent behaviors of TiniOnes!
        """

        help_label = ttk.Label(self.help_frame, text=help_text, wraplength=700, justify='left')
        help_label.pack(padx=20, pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGBugSimulatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
