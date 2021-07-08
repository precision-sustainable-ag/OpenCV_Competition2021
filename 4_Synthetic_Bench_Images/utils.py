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
    background, pot_loc = overlay_pot(background, pot_paths,rand_seed, pot_alignment)

    new_background, mask = overlay_plant(background, plant_paths, pot_loc)
    return new_background , mask

def overlay_pot(background, pot_paths, rand_seed, pot_alignment):
    """
    Returns BGRA image and single channel mask
    https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
    """
    
    background_width, background_height = background.shape[1], background.shape[0]
    
    pot_num = 0
    for pot_path in pot_paths:
        pot_foreground = load_rgba_img(pot_path) 
        
        # Random rotaton
        random.seed(rand_seed)
        rot = random.randint(0,359)
        pot_foreground = ndimage.rotate(pot_foreground, rot, reshape=True)
        
        # Get potpostions
        rndm_seed = 42
        # all_pot_pos = pot_positions(pot_foreground, rndm_seed, pot_alignment=6)
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
            # y = abs(y)
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

        # foreground foreground and create mask
        pot_foreground_image = pot_foreground[..., :3]
        pot_mask = pot_foreground[..., 3:] / 255
        # Paste foreground on background using mask
        background[y:y+h, x:x+w] = (1.0 - pot_mask) * background[y:y+h, x:x+w] + pot_mask * pot_foreground_image  
        pot_num += 1
    return background, all_pot_pos

def overlay_plant(background, pot_paths, pot_position):
    
    background_width, background_height = background.shape[1], background.shape[0]
    new_mask = np.zeros_like(background)
    pot_num = 0
    for pot_path in pot_paths:
        species = Path(pot_path).parts[-2]
        pot_foreground = load_rgba_img(pot_path) 
        
        # Random rotaton
        rot = random.randint(0,359)
        pot_foreground = ndimage.rotate(pot_foreground, rot, reshape=True)
        
        # Get potpostions
        xy = plant_positions(pot_position, species)[pot_num]
        # xy = pot_position[pot_num]
        x, y, = xy[0], xy[1]

        if x >= background_width or y >= background_height:
            return background
        
        # Get foreground background info
        h, w = pot_foreground.shape[0], pot_foreground.shape[1]
        if (h*w) > 100000:
            pot_foreground = cv2.resize(pot_foreground, (int(h/1.5), int(w/1.5)))
            h, w = pot_foreground.shape[0], pot_foreground.shape[1]

        # Crops foreground and resets values if x position is negative
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

        # foreground foreground and create mask
        pot_foreground_image = pot_foreground[..., :3]    
        pot_mask = pot_foreground[..., 3:] / 255
        # Paste foreground on background using mask
        background[y:y+h, x:x+w] = (1.0 - pot_mask) * background[y:y+h, x:x+w] + pot_mask * pot_foreground_image 
        pot_num += 1
        # Map new mask
        spec_mp = species_map(species)
        
        # pot_mask = cv2.cvtColor(pot_mask.astype('uint8'), cv2.COLOR_BGR2BGRA)

        new_mask[y:y+h, x:x+w] = (spec_mp) * pot_mask + new_mask[y:y+h, x:x+w]*(1.0-pot_mask)
        

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

def plant_positions( pot_positions, species):
    adjx = 0
    adjy = 0

    if len(pot_positions) ==9:
        if species in ['clover', 'horseweed']:
            adjx = random.randint(50,100)
            adjy = random.randint(50,100)

        if species in ['sunflower', 'velvetleaf', 'goosefoot']:
            adjx = random.randint(5,15)
            adjy = random.randint(5,15)
        
        pt1 = ( pot_positions[0][0] + adjx , pot_positions[0][1] + adjy )
        pt2 = ( pot_positions[1][0] + adjx , pot_positions[1][1] + adjy )
        pt3 = ( pot_positions[2][0] + adjx , pot_positions[2][1] + adjy )
        
        pt4 = ( pot_positions[3][0] + adjx , pot_positions[3][1] + adjy )
        pt5 = ( pot_positions[4][0] + adjx , pot_positions[4][1] + adjy )
        pt6 = ( pot_positions[5][0] + adjx , pot_positions[5][1] + adjy )
        
        pt7 = ( pot_positions[6][0] + adjx , pot_positions[6][1] + adjy )
        pt8 = ( pot_positions[7][0] + adjx , pot_positions[7][1] + adjy )
        pt9 = ( pot_positions[8][0] + adjx , pot_positions[8][1] + adjy )
        
        pot_locs = [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9]
        
    if len(pot_positions) ==6:
        if species in ['clover', 'horseweed']:
            adjx = random.randint(50,100)
            adjy = random.randint(50,100)
        if species in ['sunflower', 'velvetleaf', 'goosefoot']:
            adjx = random.randint(5,15)
            adjy = random.randint(5,15)
        
        pt1 = ( pot_positions[0][0] + adjx , pot_positions[0][1] + adjy )
        pt2 = ( pot_positions[1][0] + adjx , pot_positions[1][1] + adjy )
        pt3 = ( pot_positions[2][0] + adjx , pot_positions[2][1] + adjy )
        pt4 = ( pot_positions[3][0] + adjx , pot_positions[3][1] + adjy )
        pt5 = ( pot_positions[4][0] + adjx , pot_positions[4][1] + adjy )
        pt6 = ( pot_positions[5][0] + adjx , pot_positions[5][1] + adjy )
        
        pot_locs = [pt1, pt2, pt3, pt4, pt5, pt6]
    
    if len(pot_positions)  == 3:
        if species in ['clover', 'horseweed']:
            adjx = random.randint(5,10)
            adjy = random.randint(10,20)
        
        if species in ['sunflower', 'velvetleaf', 'goosefoot']:
            adjx = random.randint(-15,15)
            adjy = random.randint(5,15)
        
        # Align three pots down the middle
        pt1 = ( pot_positions[0][0] + adjx , pot_positions[0][1] + adjy )
        pt2 = ( pot_positions[1][0] + adjx , pot_positions[1][1] + adjy )
        pt3 = ( pot_positions[2][0] + adjx , pot_positions[2][1] + adjy )
        
        pot_locs = [pt1, pt2, pt3]
    return pot_locs