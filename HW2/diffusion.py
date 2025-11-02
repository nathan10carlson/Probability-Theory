import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # <- crucial for PyCharm GUI

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Hyperparameters ---
num_steps = 1000
batch_size = 128
lr = 1e-3
epochs = 1000


# --- Create dataset: points forming a circle ---
def sample_circle(batch_size):
    angles = tf.random.uniform([batch_size], 0, 2 * np.pi)
    x = tf.cos(angles)
    y = tf.sin(angles)
    points = tf.stack([x, y], axis=1)
    return points


# --- Model ---
class DiffusionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(2)  # Predict noise in 2D

    def call(self, x, t):
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)  # (batch, 1, features)

        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        t_float = tf.cast(t, tf.float32) / num_steps
        t_embed = tf.expand_dims(t_float, axis=1)  # (batch,1)
        t_embed_repeated = tf.repeat(t_embed, seq_len, axis=1)  # (batch, seq_len)
        t_embed_expanded = tf.expand_dims(t_embed_repeated, axis=-1)  # (batch, seq_len, 1)

        h = tf.concat([x, t_embed_expanded], axis=-1)
        h = tf.reshape(h, [batch_size * seq_len, -1])

        h = self.dense1(h)
        h = self.dense2(h)
        output = self.dense3(h)
        output = tf.reshape(output, [batch_size, seq_len, -1])

        return tf.squeeze(output, axis=1)


# --- Diffusion process ---
beta_start = 1e-4
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)
alphas = 1 - betas
alpha_cumprod = np.cumprod(alphas, axis=0)


def q_sample(x0, t, noise=None):
    if noise is None:
        noise = tf.random.normal(shape=tf.shape(x0))
    sqrt_alpha_cumprod = tf.sqrt(alpha_cumprod[t])
    sqrt_one_minus_alpha = tf.sqrt(1 - alpha_cumprod[t])
    return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise


def p_sample(model, x_t, t):
    t_tensor = tf.fill([tf.shape(x_t)[0]], t)
    pred_noise = model(x_t, t_tensor)
    alpha = alphas[t]
    alpha_cum = alpha_cumprod[t]
    beta = betas[t]
    sqrt_one_minus_alpha_cumprod = np.sqrt(1 - alpha_cum)

    coef1 = 1 / tf.sqrt(alpha)
    coef2 = beta / sqrt_one_minus_alpha_cumprod
    mean = coef1 * (x_t - coef2 * pred_noise)

    if t > 0:
        noise = tf.random.normal(tf.shape(x_t))
        sigma = np.sqrt(beta)
        sample = mean + sigma * noise
    else:
        sample = mean
    return sample


# --- Training ---
model = DiffusionModel()
optimizer = tf.keras.optimizers.Adam(lr)


@tf.function
def train_step(x0):
    batch_size = tf.shape(x0)[0]
    t = tf.random.uniform([batch_size], minval=0, maxval=num_steps, dtype=tf.int32)
    noise = tf.random.normal(tf.shape(x0))
    sqrt_alpha_cumprod_t = tf.gather(tf.sqrt(alpha_cumprod), t)
    sqrt_one_minus_alpha_cumprod_t = tf.gather(tf.sqrt(1 - alpha_cumprod), t)
    x_t = sqrt_alpha_cumprod_t[:, None] * x0 + sqrt_one_minus_alpha_cumprod_t[:, None] * noise

    with tf.GradientTape() as tape:
        pred_noise = model(x_t, t)
        loss = tf.reduce_mean(tf.square(noise - pred_noise))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# --- Generate animation frames for entire batch ---
def generate_animation_frames(x0_batch):
    frames = []
    batch = x0_batch  # shape (batch_size, 2)
    x_t = batch

    # Forward diffusion (add noise gradually)
    for t in range(num_steps):
        x_t = q_sample(batch, t)
        frames.append(x_t.numpy())

    # Reverse diffusion (denoise gradually)
    for t in reversed(range(num_steps)):
        x_t = p_sample(model, x_t, t)
        frames.append(x_t.numpy())

    return frames


# --- Main ---
if __name__ == "__main__":
    x_train = sample_circle(batch_size)

    # Training loop
    for epoch in range(epochs):
        loss = train_step(x_train)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

    model.save_weights("diffusion_model.weights.h5")

    # Animate diffusion on a new batch of points
    x_start = sample_circle(batch_size)
    frames = generate_animation_frames(x_start)

    fig, ax = plt.subplots()
    scat = ax.scatter([], [])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title("Diffusion and Reverse Diffusion Animation")

    # Add text annotation for timestep
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


    def update(frame_idx):
        scat.set_offsets(frames[frame_idx])
        # Show current timestep (frame index)
        # Because frames contains 2*num_steps frames, we can show forward or reverse step
        total_frames = len(frames)
        if frame_idx < num_steps:
            t = frame_idx  # Forward diffusion step
            phase = "Forward"
        else:
            t = total_frames - frame_idx - 1  # Reverse diffusion step
            phase = "Reverse"
        time_text.set_text(f"Timestep: {t} ({phase})")
        return scat, time_text


    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)
    plt.show()