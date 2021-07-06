import pygame
import torch
import numpy as np

from vae import load_network


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

ssize = 856
size = (ssize, ssize)
screen = pygame.display.set_mode(size)

pygame.init()
clock = pygame.time.Clock()

font = pygame.font.SysFont("Segoe UI", 30)

framerate = 60

done = False

model_name = "control_T1_5_250"
is_RS = False

model, _ = load_network(model_name)

if torch.cuda.is_available():
    model.cuda()

model.eval()

if is_RS:
    grid_size = 264
else:
    grid_size = int(np.ceil(np.sqrt(model.input_size)))
square_size = int(ssize / (grid_size))

started = False

i, j = 0, 0

curr_latent = 0
curr_val = 0
prev_latent = curr_latent
prev_val = curr_val 

curr_z = np.zeros(model.latent_size)
prev_z = curr_z.copy()

diff = np.zeros(model.input_size)
diff_norm = np.zeros(model.input_size)

color_map = lambda x: (x, 0, 255 - x)

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                curr_z = np.zeros(model.latent_size)
                prev_z = curr_z.copy()
                diff = np.zeros(model.input_size)
                diff_norm = np.zeros(model.input_size)
            if event.key == pygame.K_DOWN:
                curr_latent = (curr_latent - 1) % int(model.latent_size)
            elif event.key == pygame.K_UP:
                curr_latent = (curr_latent + 1) % int(model.latent_size)
    
    keys=pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        curr_val += 0.05
    elif keys[pygame.K_LEFT]:
        curr_val -= 0.05
        
    if curr_latent == prev_latent:
        if curr_val != prev_val:
            curr_z[curr_latent] = curr_val
            
            curr_z_torch = torch.from_numpy(curr_z).float()
            prev_z_torch = torch.from_numpy(prev_z).float()
            
            if torch.cuda.is_available():
                curr_z_torch = curr_z_torch.cuda()
                prev_z_torch = prev_z_torch.cuda()
                
            curr_pred = model.decode(curr_z_torch).data.cpu().numpy()
            prev_pred = model.decode(prev_z_torch).data.cpu().numpy()
            curr_diff = np.abs(curr_pred - prev_pred)
            prev_mul = 0.95
            # Latent difference sum update rule
            diff = curr_diff + prev_mul*diff
            diff_norm = diff / diff.max()
            if is_RS:
                diff_norm_tmp = np.tril(diff_norm) + np.tril(diff_norm, -1).T
                diff_norm = diff_norm_tmp.ravel()
    else:
        curr_val = 0
            
    prev_z = curr_z.copy()
    prev_latent = curr_latent
    prev_val = curr_val
    
    screen.fill(BLACK)
    
    for i in range(grid_size):
        for j in range(grid_size):
            if i*grid_size + j >= model.input_size:
                break
            l = j * square_size
            t = i * square_size
            w = square_size
            h = square_size
            
            if not np.isnan(diff_norm[i*grid_size + j] * 255):
                color = color_map(int(diff_norm[i*grid_size + j] * 255))
                pygame.draw.rect(screen, color, (l, t, w, h))
                
            pygame.draw.rect(screen, WHITE, (l, t, w, h), 2)
            
            latentnumtext = font.render("Latent #: {}".format(curr_latent), True, WHITE)
            latentvaltext = font.render("Latent Value: {:.2f}".format(curr_val), True, WHITE)
            screen.blit(latentnumtext, (30, ssize - latentnumtext.get_height() - 20))
            screen.blit(latentvaltext, (ssize - latentvaltext.get_width() - 30, ssize - latentvaltext.get_height() - 20))
            
    pygame.display.flip()
    clock.tick(framerate)
    
    started = True
    
pygame.quit()