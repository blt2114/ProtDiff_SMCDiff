"""Diffusion methods as described in Denoising Diffusion Probabilistic Models.
"""
import numpy as np
import torch


class Diffuser:

    def __init__(self,
                 T=512,
                 b_0=0.0001,
                 b_T=0.02,
                 ):
        """
        Args:
            T: length of diffusion.
            b_0: starting value in variance schedule.
            b_T: ending value in variance schedule.
            jax_mode: whether to run with jnp instead of np.
        """
        self.T = T
        self.b_0 = b_0
        self.b_T = b_T
        self.b_schedule = np.linspace(b_0, b_T, T)
        self.a_schedule = 1 - self.b_schedule
        self.cum_a_schedule = np.cumprod(self.a_schedule)

    def ar_forward_diffusion(self, x_t_1, t, noise_fn=None, key=None):
        b_t = self.b_schedule[t]
        noise_t = self.sample_normal(scale=1., size=x_t_1.shape, key=key)
        if noise_fn is not None:
            noise_t = noise_fn(noise_t)
        x_t = np.sqrt(1 - b_t) * x_t_1 + np.sqrt(b_t) * noise_t
        return x_t, noise_t

    def closed_form_forward_diffuse(self, x_0, t, noise_fn=None, key=None):
        cum_a_t = self.cum_a_schedule[t]
        if hasattr(t, "shape") and len(t.shape)>0:
            noise_t = self.sample_normal(size=t.shape + x_0.shape, key=key)
            if noise_fn is not None:
                noise_t = noise_fn(noise_t)
            x_t = np.sqrt(cum_a_t)[:, None, None] * x_0[None] + \
                    np.sqrt(1 - cum_a_t)[:, None, None] * noise_t
        else:
            noise_t = self.sample_normal(size=x_0.shape, key=key)
            if noise_fn is not None:
                noise_t = noise_fn(noise_t)
            x_t = np.sqrt(cum_a_t) * x_0 + np.sqrt(1 - cum_a_t) * noise_t
        return x_t, noise_t

    def sample_normal(self, size, scale=1.0, key=None):
        return np.random.normal(scale=scale, size=size)

    def ar_reverse_diffusion(
            self, x_t, e_t, t, noise_scale=1, z=None, mask=None, key=None):
        """ar_reverse_diffusion samples previous step of the diffusion

        Args:
            noise_scale: multiplicative factor on noise term, seting to less can 1 can improve sample ideality
            z: isotropic noise variable to add into the reverse diffusion sampling step.

        Returns:
            samled previous next time step
        """
        b_t = self.b_schedule[t]
        a_t = self.a_schedule[t]
        cum_a_t = self.cum_a_schedule[t]
        pred_noise = (1 - a_t) / np.sqrt(1 - cum_a_t) * e_t
        if z is None and t > 0:
            z = self.sample_normal(size=x_t.shape, key=key)
        elif z is None:
            z = 0.0
        x_t_1 = 1 / np.sqrt(a_t) * (x_t - pred_noise) + z * np.sqrt(b_t) * noise_scale
        if mask is not None:
            x_t_1 = x_t_1 * mask + (1 - mask) * x_t
        return x_t_1

    def ar_reverse_diffusion_distribution(
            self, x_t, e_t, t, mask=None):
        b_t = self.b_schedule[t]
        a_t = self.a_schedule[t]
        cum_a_t = self.cum_a_schedule[t]
        pred_noise = (1 - a_t) / np.sqrt(1 - cum_a_t) * e_t
        mu_t_1 = 1 / np.sqrt(a_t) * (x_t - pred_noise)
        sd_t_1 = np.sqrt(b_t)
        if mask is not None:
            mu_t_1 = mu_t_1 * mask
        return mu_t_1, sd_t_1


    def closed_form_reverse_diffuse(self, x_t, e_t, t):
        cum_a_t = self.cum_a_schedule[t]
        x_0 = (x_t - np.sqrt(1 - cum_a_t) * e_t) / np.sqrt(cum_a_t)
        return x_0
