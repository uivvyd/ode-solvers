import torch
import numpy as np

class VEDiffusion:
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

    def get_timesteps(self, sigma_min, sigma_max, num_steps, rho):
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        return t_steps
    
    def get_velocity_from_denoiser(self, x, model, sigma, **model_kwargs):
        sigma = sigma[:, None, None, None]
        denoised = model(x, sigma, **model_kwargs)
        velocity = (x - denoised) / sigma
        return velocity


class EulerSolver(VEDiffusion):
    def __init__(self, *args, **kwargs):
        super(EulerSolver, self).__init__(*args, **kwargs)

    def sample(self, net, noise, sigma_min=0.002, sigma_max=80.0, num_steps=20, rho=7.0):
        t_steps = self.get_timesteps(sigma_min, sigma_max, num_steps, rho)
        x = (noise * sigma_max).to(self.device)
        x_history = [x]
        with torch.no_grad():
            for i in range(len(t_steps) - 1):
                t_cur = t_steps[i]
                t_next = t_steps[i + 1]
                t_net = t_steps[i] * torch.ones(x.shape[0], device=self.device)
                delta_t = t_next - t_cur

                velocity_cur = self.get_velocity_from_denoiser(x, net, t_net)
                x = x + velocity_cur * delta_t
                x_history.append(x)

        return x, x_history


class EDM(VEDiffusion):
    def __init__(self, *args, **kwargs):
        super(EDM, self).__init__(*args, **kwargs)

    def sample(self, net, noise, sigma_min=0.002, sigma_max=80.0, num_steps=20, rho=7.0):
        t_steps = self.get_timesteps(sigma_min, sigma_max, num_steps, rho)
        x = (noise * sigma_max).to(self.device)
        x_history = [x]
        with torch.no_grad():
            for i in range(len(t_steps) - 1):
                t_cur = t_steps[i]
                t_next = t_steps[i + 1]
                t_net = t_steps[i] * torch.ones(x.shape[0], device=self.device)
                delta_t = t_next - t_cur

                velocity_cur = self.get_velocity_from_denoiser(x, net, t_net)
                x_hat = x + velocity_cur * delta_t

                if i < num_steps - 1:
                    t_next_net = t_next * torch.ones(x.shape[0], device=self.device)
                    velocity_hat = self.get_velocity_from_denoiser(x_hat, net, t_next_net)
                    x = x + 0.5 * (velocity_cur + velocity_hat) * delta_t

                x_history.append(x)
        return x, x_history
    

class VPDiffusion:
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

    def get_named_beta_schedule(self, schedule_name, beta_min=1e-4, beta_max=1e-2, num_steps=1000):
        scale = 1000 / num_steps
        beta_min *= scale
        beta_max *= scale
        if schedule_name == "linear":
            return torch.linspace(beta_min, beta_max, num_steps, dtype=torch.float64)
        elif schedule_name == "quad":
            return torch.linspace(beta_min**0.5, beta_max**0.5, num_steps) ** 2
        elif schedule_name == "sigmoid":
            betas = torch.linspace(-6, 6, num_steps)
            betas = torch.sigmoid(betas) * (beta_max - beta_min) + beta_min
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
        
    def get_alphas(self, beta_min=1e-4, beta_max=1e-2, num_steps=1000, beta_schedule="linear"):
        betas = self.get_named_beta_schedule(beta_schedule, beta_min, beta_max, num_steps)
        
        alphas = 1 - betas
        alphas_hat = torch.cumprod(alphas, dim=0)
        return {
            "betas": betas,
            "alphas": alphas,
            "alphas_hat": alphas_hat
        }
    
    def get_noise_from_denoiser(self, x, model, alpha, **model_kwargs):
        alpha = alpha[:, None, None, None]
        sigmas_ve = torch.sqrt((1 - alpha) / alpha)
        denoised_ve = model(x / torch.sqrt(alpha), sigmas_ve, **model_kwargs)
        noise_vp = (x - torch.sqrt(alpha) * denoised_ve) / torch.sqrt(1 - alpha)
        return noise_vp
    

class DDIMSolver(VPDiffusion):
    def __init__(self, *args, **kwargs):
        super(DDIMSolver, self).__init__(*args, **kwargs)

    def sample(self, net, noise, beta_start=1e-4, beta_end=1e-2, num_steps=20, beta_schedule="linear", steps_schedule="all", max_steps=20):
        if steps_schedule == "all":
            timestamps = np.arange(num_steps)
        elif steps_schedule == 'linear':
            step_size = num_steps // max_steps
            timestamps = np.arange(0, num_steps, step_size)[:max_steps]
        elif steps_schedule == "quad":
            timestamps = (np.linspace(0, np.sqrt(num_steps) - 1, max_steps) ** 2).astype(int)
        else:
            raise NotImplementedError(f"unknown steps schedule: {steps_schedule}")
        
        alphas_dict = self.get_alphas(beta_start, beta_end, num_steps, beta_schedule)
        alphas = alphas_dict["alphas_hat"][timestamps]
        alphas = torch.flip(alphas, dims=(0, ))

        x = (noise).to(self.device)
        x_history = [x]
        with torch.no_grad():
            for i in range(len(alphas) - 1):
                alpha_cur = alphas[i]
                alpha_prev = alphas[i + 1]
                alpha_net = alpha_cur * torch.ones(x.shape[0], device=self.device)

                eps = self.get_noise_from_denoiser(x, net, alpha_net)
                predicted_x0 = (x - torch.sqrt(1 - alpha_cur) * eps) / torch.sqrt(alpha_cur)
                direction_xt = torch.sqrt(1 - alpha_prev) * eps
                x = torch.sqrt(alpha_prev) * predicted_x0 + direction_xt
                x_history.append(x)

        return x, x_history
    

class DPMSolver1(VPDiffusion):
    def __init__(self, *args, **kwargs):
        super(DPMSolver1, self).__init__(*args, **kwargs)

    
    def step(self, ind, x, net):
        h = self.lmbd[ind+1] - self.lmbd[ind]
        alpha_t = self.alphas[ind] * torch.ones(x.shape[0], device=self.device)
        pred_noise = self.get_noise_from_denoiser(x, net, alpha_t)
        return self.mean[ind+1] / self.mean[ind] * x - self.sigma[ind+1] * torch.expm1(h) * pred_noise
        

    def sample(self, net, noise, eps=1e-3, T=1, beta_0=0.1, beta_1=20, num_steps=20):
        timestamps = torch.linspace(eps, T, num_steps) 
        alphas = torch.exp(-(beta_1 - beta_0) / 4 * timestamps ** 2 - beta_0 / 2 * timestamps)
        self.alphas = torch.flip(alphas, dims=(0,))

        self.mean = torch.sqrt(self.alphas)
        self.sigma = torch.sqrt(1 - self.alphas)
        self.lmbd = torch.log(self.mean / self.sigma)
        x = noise.to(self.device)
        x_history = [x]
        with torch.no_grad():
            for i in range(0, len(alphas)-1): 
                x = self.step(i, x, net)
                x_history.append(x)

        return x, x_history
