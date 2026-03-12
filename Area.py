import numpy as np

from included_functions_Area import *
import placentagen as pg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sample_number = 'JT26008'
placenta_type = 'FGR'
img_input_dir = placenta_type + '/'


if not os.path.exists(img_input_dir):
    print('incorrect input directory. Given input directory is ' + img_input_dir )


###############################################################
# ---------------- Set DEBUG Variables ---------------------- #
###############################################################

use_custom_pixel_scale = True
show_debug_image = True
###############################################################
# Parameters
###############################################################/
scalebar_size = 10 #mm
pixel_scale = 0.1176
#######################################################################
#------------------------Scale Generation-----------------------------#
#######################################################################


if use_custom_pixel_scale:
    print('Using Custom Pixel Scale')
    scale_filename = img_input_dir + sample_number + '_scale.png'
    scale_file = read_png(scale_filename, 'g')
    pixel_scale = get_scale(10, scale_file)
else:
    print('Using default pixel scale')
print('Scale: ' + str(pixel_scale) + ' mm/pixel')

#read placenta outline
placenta_area_filename = sample_number + '_area.png'
green_pixels, placenta_area = calculate_area(img_input_dir+placenta_area_filename,pixel_scale, show_debug_image)
print(f"Number of Green Pixels: {green_pixels}")
print(f"Area in mm²: {placenta_area:.2f}")