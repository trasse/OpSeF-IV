# Settings for Patch Extractor

def config_patch_extractor():
    image_formats = {}

# Slide Scanner no scale
    pE = {}
    pE["step_x"] = 530
    pE["step_y"] = 530
    pE["top_left"] = [18,18] # (y,x)
    pE["dim_x"] = 3
    pE["dim_y"] = 3
    pE["patch_y"] = 512
    pE["patch_x"] = 512
    pE["ScaleDown"] = 1 # make smaller patches for faster classifier results
    pE["Z_Project"] = False
    image_formats["S1"] = pE

# Slide Scanner Scale 2
    pE = {}
    pE["step_x"] = 0
    pE["step_y"] = 0
    pE["top_left"] = [292,292] # (y,x)
    pE["dim_x"] = 1
    pE["dim_y"] = 1
    pE["patch_y"] = 1024
    pE["patch_x"] = 1024
    pE["ScaleDown"] = 2 # make smaller patches for faster classifier results
    pE["Z_Project"] = False
    image_formats["S2"] = pE

# Slide Scanner Scale 3
    pE = {}
    pE["step_x"] = 0
    pE["step_y"] = 0
    pE["top_left"] = [36,36] # (y,x)
    pE["dim_x"] = 1
    pE["dim_y"] = 1
    pE["patch_y"] = 1536
    pE["patch_x"] = 1536
    pE["ScaleDown"] = 3 # make smaller patches for faster classifier results
    pE["Z_Project"] = False
    image_formats["S3"] = pE

# Confocal Stack 1024
    pE = {}
    pE["step_x"] = 512
    pE["step_y"] = 512
    pE["top_left"] = [0,0] # (y,x)
    pE["dim_x"] = 2
    pE["dim_y"] = 2
    pE["patch_y"] = 512
    pE["patch_x"] = 512
    pE["ScaleDown"] = 1 # make smaller patches for faster classifier results
    pE["Z_Project"] = True
    image_formats["C1"] = pE

# Confocal Stack 1536
    pE = {}
    pE["step_x"] = 512
    pE["step_y"] = 512
    pE["top_left"] = [0,0] # (y,x)
    pE["dim_x"] = 3
    pE["dim_y"] = 3
    pE["patch_y"] = 512
    pE["patch_x"] = 512
    pE["ScaleDown"] = 1 # make smaller patches for faster classifier results
    pE["Z_Project"] = True
    image_formats["C2"] = pE

# Confocal Stack 2048
    pE = {}
    pE["step_x"] = 512
    pE["step_y"] = 512
    pE["top_left"] = [0,0] # (y,x)
    pE["dim_x"] = 4
    pE["dim_y"] = 4
    pE["patch_y"] = 512
    pE["patch_x"] = 512
    pE["ScaleDown"] = 1 # make smaller patches for faster classifier results
    pE["Z_Project"] = True
    image_formats["C3"] = pE

# Confocal Z-Proj 1024
    pE = {}
    pE["step_x"] = 512
    pE["step_y"] = 512
    pE["top_left"] = [0,0] # (y,x)
    pE["dim_x"] = 2
    pE["dim_y"] = 2
    pE["patch_y"] = 512
    pE["patch_x"] = 512
    pE["ScaleDown"] = 1 # make smaller patches for faster classifier results
    pE["Z_Project"] = False
    image_formats["Z1"] = pE

# Confocal Z-Proj 1536
    pE = {}
    pE["step_x"] = 512
    pE["step_y"] = 512
    pE["top_left"] = [0,0] # (y,x)
    pE["dim_x"] = 3
    pE["dim_y"] = 3
    pE["patch_y"] = 512
    pE["patch_x"] = 512
    pE["ScaleDown"] = 1 # make smaller patches for faster classifier results
    pE["Z_Project"] = False
    image_formats["Z2"] = pE

# Confocal Z-Proj 2048
    pE = {}
    pE["step_x"] = 512
    pE["step_y"] = 512
    pE["top_left"] = [0,0] # (y,x)
    pE["dim_x"] = 4
    pE["dim_y"] = 4
    pE["patch_y"] = 512
    pE["patch_x"] = 512
    pE["ScaleDown"] = 1 # make smaller patches for faster classifier results
    pE["Z_Project"] = False
    image_formats["Z3"] = pE
    return image_formats