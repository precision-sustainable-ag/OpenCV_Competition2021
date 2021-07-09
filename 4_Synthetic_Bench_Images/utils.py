import random
import numpy as np
from scipy import ndimage
import cv2
from pathlib import Path


def load_rgba_img(image_path):
    # Load images
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    bgra_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    return bgra_img


def overlay(bench_path, pot_paths, plant_paths,pot_alignment): # 'pot' or 'plant'
    
    rand_seed = random.seed(42)
    # load individual images
    background = cv2.imread(str(bench_path))
    # background_width, background_height = background.shape[1], background.shape[0]
    background, all_pot_pos, pot_hw = overlay_pot(background, pot_paths,rand_seed, pot_alignment)

    new_background, mask = overlay_plant(background, plant_paths, all_pot_pos, pot_hw)

    return new_background , mask

def overlay_pot(background, pot_paths, rand_seed, pot_alignment):
    """
    Returns BGRA image and single channel mask
    """
    
    background_width, background_height = background.shape[1], background.shape[0]
    pot_heights_widths = []
    pot_num = 0
    for pot_path in pot_paths:
        pot_foreground = load_rgba_img(pot_path) 
        
        # Random rotaton
        random.seed(rand_seed)
        rot = random.randint(0,359)
        pot_foreground = ndimage.rotate(pot_foreground, rot, reshape=True)
        
        # Get potpostions
        rndm_seed = 42
        if pot_num != 0:
            pass
        else:
            # Get potpostions
            all_pot_pos = pot_positions(pot_foreground, rndm_seed, pot_alignment=pot_alignment)
        pot_pos = all_pot_pos[pot_num]
        x, y = pot_pos[0], pot_pos[1]

        if x >= background_width or y >= background_height:
            return background
        
        # Get foreground background info
        h, w = pot_foreground.shape[0], pot_foreground.shape[1]
   
        if x <= 0:
            pot_foreground = pot_foreground[:,int(w/2 + x):]
            h, w = pot_foreground.shape[0], pot_foreground.shape[1]
            x = 0
        if y <= 0:
            pot_foreground = pot_foreground[int(h/2 + y):,:]
            h, w = pot_foreground.shape[0], pot_foreground.shape[1]
            y = 0

        if x + w > background_width:
            w = background_width - x
            pot_foreground = pot_foreground[:, :w]
        
        if y + h > background_height:
            h = background_height - y
            pot_foreground = pot_foreground[:h]
        # Make RGBA if necessary
        if pot_foreground.shape[2] < 4:
            pot_foreground = np.concatenate(
            [
            pot_foreground,
            np.ones((pot_foreground.shape[0], pot_foreground.shape[1], 1), dtype = pot_foreground.dtype) * 255
            ],
            axis = 2,
        )

        h, w = pot_foreground.shape[0], pot_foreground.shape[1]
        pot_heights_widths.append(tuple((h,w)))
        # foreground foreground and create mask
        pot_foreground_image = pot_foreground[..., :3]
        pot_mask = pot_foreground[..., 3:] / 255
        # Paste foreground on background using mask
        background[y:y+h, x:x+w] = (1.0 - pot_mask) * background[y:y+h, x:x+w] + pot_mask * pot_foreground_image  
        pot_num += 1
        
    return background, all_pot_pos, pot_heights_widths

def overlay_plant(background, plant_paths, all_pot_pos,pot_hw):
    
    background_width, background_height = background.shape[1], background.shape[0]
    new_mask = np.zeros_like(background)
    pot_num = 0
    for plant_path in plant_paths:
        species = Path(plant_path).parts[-2]
        plant_foreground = load_rgba_img(plant_path) 
        
        # Random rotaton
        rot = random.randint(0,359)
        plant_foreground = ndimage.rotate(plant_foreground, rot, reshape=True)
        # Get foreground background info
        h, w = plant_foreground.shape[0], plant_foreground.shape[1]

        # Get potpostions
        pot_center = pot_centers(all_pot_pos, pot_hw)[pot_num]
        
        x, y = plant_location(pot_center, pot_num, plant_foreground.shape)
        # xy = pot_position[pot_num]
        # x, y, = pot_center[0] - int(w/2), pot_center[1] - int(h/2)

        if x >= background_width or y >= background_height:
            return background
        
        if (h*w) > 100000:
            plant_foreground = cv2.resize(plant_foreground, (int(h/1.5), int(w/1.5)))
            h, w = plant_foreground.shape[0], plant_foreground.shape[1]

        # Crops foreground and resets values if x position is negative
        if x <= 0:
            plant_foreground = plant_foreground[:,int(w/2 + x):]
            h, w = plant_foreground.shape[0], plant_foreground.shape[1]
            x = 0
        
        if y <= 0:
            plant_foreground = plant_foreground[int(h/2 + y):,:]
            h, w = plant_foreground.shape[0], plant_foreground.shape[1]
            y = 0

        if x + w > background_width:
            w = background_width - x
            plant_foreground = plant_foreground[:, :w]
        
        if y + h > background_height:
            h = background_height - y
            plant_foreground = plant_foreground[:h]
        # Make RGBA if necessary
        if plant_foreground.shape[2] < 4:
            plant_foreground = np.concatenate(
            [
            plant_foreground,
            np.ones((plant_foreground.shape[0], plant_foreground.shape[1], 1), dtype = plant_foreground.dtype) * 255
            ],
            axis = 2,
        )
        # h, w = plant_foreground.shape[0], plant_foreground.shape[1]
        # foreground foreground and create mask
        plant_foreground_image = plant_foreground[..., :3]    
        plant_mask = plant_foreground[..., 3:] / 255
        # Paste foreground on background using mask
        background[y:y+h, x:x+w] = (1.0 - plant_mask) * background[y:y+h, x:x+w] + plant_mask * plant_foreground_image 
        pot_num += 1
        # Map new mask
        spec_mp = species_map(species)
        
        # pot_mask = cv2.cvtColor(pot_mask.astype('uint8'), cv2.COLOR_BGR2BGRA)

        new_mask[y:y+h, x:x+w] = (spec_mp) * plant_mask + new_mask[y:y+h, x:x+w]*(1.0-plant_mask)
        

    return background, new_mask

def species_map(species_str):
    sp_map = {
        "soil":0,
        "clover":1,
        "cowpea":2,
        "goosefoot":3,
        "grasses":4,
        "horseweed":5,
        "sunflower":6,
        "velvetleaf":7
    }    
    res = sp_map[species_str]
    return res

def pot_positions(pot,rndm_seed, pot_alignment=6):
    """Use center of pot image for positioning. Create position xy coordinates
       for pot positioning"""
    img_h, img_w, _ = pot.shape
    center_x = int(img_w/2)
    center_y = int(img_h/2)
    
    if pot_alignment == 9:
        random.seed(rndm_seed)
        rand_x = random.randint(-75, 75)
        rand_y = random.randint(-15, 15)
        rand_y0 = random.randint(-150, 0)
    
        # Center potins for 6 quadrats of bench image
        pt1 = ( 320  - center_x + rand_x , -10 + rand_y0 )
        pt2 = ( 960  - center_x + rand_x , -10 + rand_y0  )
        pt3 = ( 1600 - center_x + rand_x , -10 + rand_y0  )
        
        pt4 = ( 320  - center_x + rand_x, 600 - center_y + rand_y  )
        pt5 = ( 960  - center_x + rand_x, 600 - center_y + rand_y  )
        pt6 = ( 1600 - center_x + rand_x, 600 - center_y + rand_y  )
        
        pt7 = ( 320  - center_x + rand_x, 1100 - center_y + rand_y  )
        pt8 = ( 960  - center_x + rand_x, 1100 - center_y + rand_y  )
        pt9 = ( 1600 - center_x + rand_x, 1100 - center_y + rand_y  )

        pot_locs = [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9]
    # Add pot center xy to new image location to get center
    if pot_alignment == 6:
        random.seed(rndm_seed)
        rand_x = random.randint(-75, 75)
        rand_y = random.randint(-15, 15)
    
        # Center potins for 6 quadrats of bench image
        pt1 = ( 320  - center_x + rand_x , 270 - center_y + rand_y )
        pt2 = ( 960  - center_x + rand_x , 270 - center_y + rand_y  )
        pt3 = ( 1600 - center_x + rand_x , 270 - center_y + rand_y  )
        pt4 = ( 320  - center_x + rand_x, 840 - center_y + rand_y  )
        pt5 = ( 960  - center_x + rand_x, 840 - center_y + rand_y  )
        pt6 = ( 1600 - center_x + rand_x, 840 - center_y + rand_y  )

        pot_locs = [pt1, pt2, pt3, pt4, pt5, pt6]
    if pot_alignment == 3:
        random.seed(rndm_seed)
        rand_x = random.randint(-10, 10)
        rand_y = random.randint(-15, 15)
    
        # Align three pots down the middle
        pt1 = (  320   - center_x + rand_x,  550 - center_y + rand_y)
        pt2 = (  960  - center_x + rand_x,  550 - center_y + rand_y)
        pt3 = (  1600 - center_x + rand_x,  550 - center_y + rand_y)
        
        pot_locs = [pt1, pt2, pt3]
    return pot_locs

def pot_centers( all_pot_pos, pot_hw):

    if len(all_pot_pos) > 0:
  
        # x, y coordinates
        pot1 = all_pot_pos[0]
        pot2 = all_pot_pos[1]
        pot3 = all_pot_pos[2]

        pot1_height = pot_hw[0][0]
        pot2_height = pot_hw[1][0] 
        pot3_height = pot_hw[2][0] 
        pot1_width  = pot_hw[0][1]
        pot2_width  = pot_hw[1][1]
        pot3_width  = pot_hw[2][1]

        pot1_center = pot1[0] + int(pot1_width/2), pot1[1] + int(pot1_height/2)
        pot2_center = pot2[0] + int(pot2_width/2), pot2[1] + int(pot2_height/2)
        pot3_center = pot3[0] + int(pot3_width/2), pot3[1] + int(pot3_height/2)
    
        plant_locs = [pot1_center, pot2_center, pot3_center]
        
    if len(all_pot_pos) > 3:
        # x, y coordinates
        pot4 = all_pot_pos[3]
        pot5 = all_pot_pos[4]
        pot6 = all_pot_pos[5]

        pot4_height = pot_hw[3][0]
        pot5_height = pot_hw[4][0] 
        pot6_height = pot_hw[5][0] 
        pot4_width  = pot_hw[3][1]
        pot5_width  = pot_hw[4][1]
        pot6_width  = pot_hw[5][1]

        pot4_center = pot4[0] + int(pot4_width/2), pot4[1] + int(pot4_height/2)
        pot5_center = pot5[0] + int(pot5_width/2), pot5[1] + int(pot5_height/2)
        pot6_center = pot6[0] + int(pot6_width/2), pot6[1] + int(pot6_height/2)

        plant_locs = [
            pot1_center, pot2_center, pot3_center, 
            pot4_center, pot5_center, pot6_center]

    if len(all_pot_pos) > 6:
        # x, y coordinates
        pot7 = all_pot_pos[6]
        pot8 = all_pot_pos[7]
        pot9 = all_pot_pos[8]

        pot7_height = pot_hw[6][0]
        pot8_height = pot_hw[7][0] 
        pot9_height = pot_hw[8][0] 
        pot7_width  = pot_hw[6][1]
        pot8_width  = pot_hw[7][1]
        pot9_width  = pot_hw[8][1]

        pot7_center = pot7[0] + int(pot7_width/2), pot7[1] + int(pot7_height/2)
        pot8_center = pot8[0] + int(pot8_width/2), pot8[1] + int(pot8_height/2)
        pot9_center = pot9[0] + int(pot9_width/2), pot9[1] + int(pot9_height/2)

        plant_locs = [
            pot1_center, pot2_center, pot3_center, 
            pot4_center, pot5_center, pot6_center,
            pot7_center, pot8_center, pot9_center]

    return plant_locs

def plant_location(pot_center, pot_number, plant_shape):
    print(pot_center)
    print(pot_number)
    # pot_center = pot_center[pot_number]

    h, w = plant_shape[0], plant_shape[1]
    x, y = pot_center[0] - int(w/2), pot_center[1] - int(h/2)

    return x, y

    
