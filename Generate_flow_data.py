import numpy as np

from included_functions_flow import *
import placentagen as pg
import matplotlib
from matplotlib.pyplot import figure

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from skan import draw, Skeleton, summarize
from reprosim.diagnostics import set_diagnostics_level
from reprosim.indices import perfusion_indices, get_ne_radius
from reprosim.geometry import append_units, define_node_geometry, define_1d_element_placenta, define_rad_from_geom, \
    add_matching_mesh, \
    define_capillary_model, define_rad_from_file
from reprosim.repro_exports import export_1d_elem_geometry, export_node_geometry, export_1d_elem_field, \
    export_node_field, export_terminal_perfusion
from reprosim.pressure_resistance_flow import evaluate_prq, calculate_stats
import csv
import os

sample_number = 'JT23070'
img_input_dir = '/media/share/derivative/2023-sex-specific/chorionic-segmentations/' +sample_number +'/'
output_tree_dir = 'outputs_grow_tree/' + sample_number + '/'
output_flow_dir = 'outputs_flow_tree/' + sample_number + '/'
output_table_dir = 'outputs_branch_stats/' + sample_number + '/'
if not os.path.exists(output_tree_dir):
    os.makedirs(output_tree_dir)
if not os.path.exists(output_flow_dir):
    os.makedirs(output_flow_dir)
if not os.path.exists(output_table_dir):
    os.makedirs(output_table_dir)

###############################################################
# ---------------- Set DEBUG Variables ---------------------- #
###############################################################

use_custom_pixel_scale = True
debug_export_all = True
show_debug_images = False
inlet_type = 'double'
inlet_node = True
is_rotate = False
is_rotated = False
###############################################################
# Parameters that define branching within the placenta volume #
###############################################################/
#Number of seed points targeted for growing tree
n_seed = 32000
Reference_volume = 427700
#Maximum angle between two branches
angle_max_ft = 100 * np.pi / 180
#Minimum angle between two branches
angle_min_ft = 0 * np.pi / 180
#Fraction that the branch grows toward data group centre of mass at each iteration
fraction_ft = 0.4
#Minimum length of a branch
min_length_ft = 1.0  #mm
#minimum number of data points that can be in any group after a data splitting proceedure
point_limit_ft = 1
#pixel density
pixel_scale = 0.04 #mm/pixel
#placenta measurements
thickness = 20 #mm
t_pixels = int(thickness / pixel_scale)
t_half = int(t_pixels * 2)
#SV and umbilical cord
sv_length = 2.0  #mm
umbilical_length = 10.0  #mm
rotation_angle = 120
scale_factor_x = 1.2
scale_factor_y = 1.2
#######################################################################
#------------------------Scale Generation-----------------------------#
#######################################################################
if use_custom_pixel_scale:
    print('Using Custom Pixel Scale')
    scale_filename = img_input_dir + sample_number + '_scale.png'
    scale_file = read_png(scale_filename, 'g')
    pixel_scale = get_scale(10, scale_file)
print('Scale: ' + str(pixel_scale) + ' mm/pixel')
#read placenta outline
placenta_area_filename = sample_number + '_area.png'
green_pixels, placenta_area = calculate_area(img_input_dir+placenta_area_filename,pixel_scale, show_debug_images)
print(f"Number of Green Pixels: {green_pixels}")
print(f"Area in mm²: {placenta_area:.2f}")
#######################################################################
#-------------------Ellipse/Hull Generation---------------------------#
#######################################################################
#read placenta outline
placenta_filename = sample_number + '_outline.png'

placenta_mask = read_png(img_input_dir + placenta_filename, 'g')

#Generate the outline of the placenta in 3D
outputfilename = output_flow_dir + sample_number + '_plac_3d'
plac_outline_nodes = generate_placenta_outline(placenta_mask, pixel_scale, thickness, outputfilename, show_debug_images,
                                               debug_export_all, is_rotate, rotation_angle)

#Generate and export nodes that are equally spaces in the 3D spaced placental structure
filename_hull = output_tree_dir + sample_number + '_nodes'
plac_nodes = dict.fromkeys(['nodes'])
plac_nodes['nodes'] = plac_outline_nodes
datapoints, xcentre, ycentre, zcentre, volume = equispaced_data_in_hull(n_seed, plac_nodes)
if debug_export_all:
    pg.export_ex_coords(datapoints, 'placenta', filename_hull, 'exnode')
    print('Node files for placental hull generated and exported to:', filename_hull)

#Fit an ellipse to the placental outline. Weighting to bias the placenta so that more of the
#placental outline is inside the ellipse. This is to find centre point
[x, y, ellipse_fit] = fit_ellipse_2d(placenta_mask, 0.8)
x_mm = ellipse_fit[1] * pixel_scale  #x length of the placenta in mm
y_mm = ellipse_fit[0] * pixel_scale  #y length of the placenta in mm
volume = 4. * np.pi * x_mm * y_mm * (thickness / 2.) / 3.

#-------------------Transform the 3D hull ---------------------------#
# Calculate the desired center in real-world coordinates
desired_center = np.array([ellipse_fit[3] * pixel_scale, ellipse_fit[4] * pixel_scale, zcentre])

# Translate the 3D points. This is the real world coordinates used for generation
translated_points_3d = datapoints - desired_center
datapoints_ellipse, hull_params = generate_ellipse_hull(translated_points_3d)
datapoints_ellipse_array = np.array(datapoints_ellipse)

index = np.arange(datapoints_ellipse_array.shape[0]).reshape(-1, 1)  # 0-based indexing

datapoints_ellipse_array = np.hstack((index, datapoints_ellipse_array))
plac_nodes = dict.fromkeys(['nodes'])
plac_nodes['nodes'] = datapoints_ellipse_array

ellipse_hull, xcentre, ycentre, zcentre, volume = equispaced_data_in_hull(n_seed,plac_nodes)
n_seed_adjusted = int((volume*n_seed)/Reference_volume)
ellipse_hull, xcentre, ycentre,zcentre, volume = equispaced_data_in_hull(n_seed_adjusted,plac_nodes)
print('Adjusted seed points based on volume')


if debug_export_all:
    pg.export_ex_coords(translated_points_3d, 'placenta', output_tree_dir + 'villi_final_' + sample_number, 'exnode')
    pg.export_ex_coords(datapoints_ellipse_array, 'placenta', output_tree_dir + 'villi_ellipse_' + sample_number,
                        'exnode')
    print('Debug node files ellipsified hull and translated ellipse exported to :', output_tree_dir)
print('Hull Generation complete: ⸜(｡˃ ᵕ ˂ )⸝♡')

#######################################################################
#------------------- Artery tree Generation---------------------------#
#######################################################################
#arteries = read_png(img_input_dir + 'arteries_' + sample_number + '.png', 'r')
arteries = read_png(img_input_dir + sample_number + '_vessels.png', 'r')

euc_dist_image = get_euclidean_distance(arteries)

#Skeletonize the artery branches
skel_art = skeletonise_2d(arteries)

branch_data = summarize(Skeleton(skel_art, spacing=pixel_scale,source_image=arteries))
data = Skeleton(skel_art, spacing=pixel_scale,source_image=arteries)
outputfilename = output_tree_dir + 'arteries_' + sample_number
px_g, coord, art_nodes, art_elems = skel2graph(skel_art, outputfilename, debug_export_all,inlet_type,arteries)

RN,RE = get_radii_from_euclidean(art_nodes,art_elems,euc_dist_image)
pg.export_exfield_1d_linear(RE,'arteries','art_radii',output_tree_dir+'raddi_art_raw'+sample_number)
real_radii = RE *pixel_scale

if show_debug_images:
    #Analyze branch type of skeleton and plt
    #draw overlay on branch
    fig, ax = plt.subplots()
    draw.overlay_euclidean_skeleton_2d(arteries, branch_data, skeleton_color_source='branch-type')
    # Generate CS graph
    fig, ax = plt.subplots()
    display = (arteries + placenta_mask + skel_art) / 3
    ax.imshow(display)
    plt.show()
print('Chorion arteries generation complete: ৻(  •̀ ᗜ •́  ৻)')

nodes_scaled = art_nodes
nodes_scaled[:, 1] = (art_nodes[:, 1] * pixel_scale) - (ellipse_fit[3] * pixel_scale)
nodes_scaled[:, 2] = (art_nodes[:, 2] * pixel_scale) - (ellipse_fit[4] * pixel_scale)
nodes_scaled[:, 3] = max(translated_points_3d[:, 2])
if debug_export_all:
    outputfilename = output_tree_dir + 'arteries_scaled_' + sample_number
    pg.export_ex_coords(nodes_scaled, 'arteries', outputfilename, 'exnode')
    print('Arterial nodes and elems exported to: ', outputfilename)
outputfilename = output_tree_dir + 'arteries_hull_scaled_' + sample_number
arterial_shaped_nodes = map_nodes_to_hull(nodes_scaled, hull_params, thickness, outputfilename, debug_export_all)
#arterial_shaped_nodes, art_elems = pg.delete_unused_nodes(arterial_shaped_nodes, art_elems)
trees = split_trees(arterial_shaped_nodes, art_elems, real_radii)

if inlet_type == 'single':
    trees = define_geom(arterial_shaped_nodes,art_elems,real_radii)
elif inlet_type == 'double':
    trees = split_trees(arterial_shaped_nodes,art_elems,real_radii)
#arterial_shaped_nodes, art_elems = pg.delete_unused_nodes(arterial_shaped_nodes, art_elems)
Geom_A, Geom_B = chorion_branching_analytics(trees,sample_number,output_tree_dir, inlet_type,False)
if inlet_type == 'single':
    radius_inlet_branch = get_inlet_branch_radius(Geom_A)
    Geom_A = set_inlet_branch_radius(Geom_A,radius_inlet_branch)
    arterial_shaped_nodes = Geom_A['nodes']
    arterial_elems = Geom_A['elems']
    real_radii = Geom_A['radii']
elif inlet_type=='double':
    radius_inlet_branch = get_inlet_branch_radius(Geom_A)
    Geom_A = set_inlet_branch_radius(Geom_A,radius_inlet_branch)
    radius_inlet_branchB = get_inlet_branch_radius(Geom_B)
    Geom_B = set_inlet_branch_radius(Geom_B,radius_inlet_branchB)
    arterial_shaped_nodes,art_elems,real_radii = recombine_trees(Geom_A,Geom_B)
    print('Trees recombined: ⸜(｡˃ ᵕ ˂ )⸝♡⸜(｡˃ ᵕ ˂ )⸝')

outputfilename = output_tree_dir + 'Umb_' + sample_number
if inlet_node:
    nodes_Umb, elems_Umb, real_radii = create_umb_anastomosis(arterial_shaped_nodes, art_elems, umbilical_length, outputfilename,
                                                  debug_export_all, inlet_type, real_radii)
    print('Anastomosis and inlet added: ٩(^ᗜ^)و')

else:
    nodes_Umb = arterial_shaped_nodes
    elems_Umb = art_elems
    print('Anastomosis and inlet not added: ٩(^ᗜ^)و')

terminal = pg.calc_terminal_branch(nodes_Umb[:, 1:4], elems_Umb)

branch_structure, branch_data = allocate_branch_numbers(nodes_Umb, elems_Umb)
pg.export_exfield_1d_linear(branch_structure, 'arteries', 'branch', output_tree_dir + 'branch')
chorion_nodes, chorion_elems = add_stem_villi(nodes_Umb, elems_Umb, sv_length, terminal)
pg.export_exelem_1d(chorion_elems, 'arteries', output_tree_dir + 'chorion')
pg.export_ex_coords(chorion_nodes, 'arteries', output_tree_dir + 'chorion', 'exnode')
parent_list_nodes, parent_list_elems = find_parent_list(chorion_nodes, chorion_elems)

print('Chorion mapping complete: ৻(  •̀ ᗜ •́  ৻)')
#######################################################################
#----------------------- Tree Generation------------------------------#
#######################################################################
#Define new chorion and stem
chorion_and_stem_shaped = dict.fromkeys(['nodes', 'elems', 'total_nodes', 'total_elems', 'elem_up', 'elem_down'])
chorion_and_stem_shaped['nodes'] = chorion_nodes
chorion_and_stem_shaped['elems'] = chorion_elems
chorion_and_stem_shaped['total_nodes'] = len(chorion_nodes)
chorion_and_stem_shaped['total_elems'] = len(chorion_elems)
elem_cnct_shaped = pg.element_connectivity_1D(chorion_nodes[:, 1:4], chorion_elems)
chorion_and_stem_shaped['elem_up'] = elem_cnct_shaped['elem_up']
chorion_and_stem_shaped['elem_down'] = elem_cnct_shaped['elem_down']

#------------------- Tree Generation---------------------------#
#Grow tree with hull
full_geom_shaped = pg.grow_large_tree(angle_max_ft, angle_min_ft, fraction_ft, min_length_ft, point_limit_ft, volume,
                                      thickness, 0, ellipse_hull, chorion_and_stem_shaped, 1, parent_list_elems)

Tree_file = output_tree_dir + 'full_tree_' + sample_number

if is_rotated:
    full_nodes = full_geom_shaped['nodes']
    node_positions = full_nodes[:,1:4]
    inlet = node_positions[0,:]
    translated_positions = node_positions - inlet
    theta = (rotation_angle * np.pi )/ 180
    scaling_matrix = np.array([
        [scale_factor_x, 0, 0],
        [0, scale_factor_y, 0],
        [0, 0, 1]  # No scaling in z
    ])
    rotation_matrix = np.asarray([
        [np.cos(theta),np.sin(theta),0],
        [-np.sin(theta),np.cos(theta),0],
        [0,0,1]
    ])
    transformation_matrix  = scaling_matrix@rotation_matrix
    rotated_nodes = translated_positions@transformation_matrix.T
    full_nodes[:,1:4] = rotated_nodes + inlet
    full_geom_shaped['nodes'] = full_nodes
pg.export_ex_coords(full_geom_shaped['nodes'], 'placenta', Tree_file, 'exnode')
pg.export_exelem_1d(full_geom_shaped['elems'], 'placenta', Tree_file)
radii_hull_elem = pg.define_radius_by_order(full_geom_shaped['nodes'][:, 1:4], full_geom_shaped['elems'], 'strahler',
                                            0, 0.1, 1.53)
outputfilename = output_tree_dir + 'radii_' + sample_number
pg.export_exfield_1d_linear(radii_hull_elem, 'placenta', 'radii', outputfilename)
#ConvertExtoIP(Tree_file)
pg.export_ip_coords(full_geom_shaped['nodes'][:, 1:4], 'placenta', Tree_file)
pg.export_ipelem_1d(full_geom_shaped['elems'], 'placenta', Tree_file)
print('Tree generation complete: ৻(  •̀ ᗜ •́  ৻)')

volume, vessel_volumes, lengths = get_vessel_volume(full_geom_shaped['nodes'],radii_hull_elem,full_geom_shaped['elems'])
pg.calc_terminal_branch(full_geom_shaped['nodes'][:,1:4],full_geom_shaped['elems'])

####################################################################################
#----------------------------------------------------------------------------------#
#----------------------- Flow and Pressure Generation------------------------------#
#----------------------------------------------------------------------------------#
####################################################################################

print('Beginning flow and pressure simulations (ó﹏ò｡)')

###############################################################
# --------------- Flow simulation setup --------------------- #
###############################################################

set_diagnostics_level(
    0)  # level 0 - no diagnostics; level 1 - only prints subroutine names (default); level 2 - prints subroutine names and contents of variables
perfusion_indices()
#Load node points in tree
print("Reading elem file", Tree_file)
define_node_geometry(Tree_file + '.ipnode')
#Load elements in tree
define_1d_element_placenta(Tree_file + '.ipelem')

# mesh_type: can be 'simple_tree' or 'full_plus_tube'. Simple_tree is the input
## arterial tree without any special features at the terminal level
# 'full_plus_tube' creates a matching venous mesh and has arteries and
## veins connected by capillary units (capillaries are just tubes represented by an element)
mesh_type = 'full_plus_tube'
#mesh that converges onto the arterial tree
umbilical_elem_option = 'same_as_arterial'
#Boundary condition type: Needs to be either Inlet Pressure and Outlet Pressure or Outlet pressure and inlet flow rate
bc_type = 'pressure'  # 'pressure' or 'flow'
#Rheology is constant viscosity. Can also account for the effects of RBC on viscosity
rheology_type = 'constant_visc'
#Vessel type can be rigid or a elastic as a function of diameter
vessel_type = 'rigid'

# define terminal units (this subroutine always needs to be called regardless of mesh_type
append_units()

###############################################################
# --------------- Venous mesh Creation ---------------------- #
###############################################################

#venous mesh creation
umbilical_elements = []
add_matching_mesh(umbilical_elem_option, umbilical_elements)

# define radius by Strahler order in diverging (arterial mesh)
s_ratio = 1.38  # rate of decrease in radius at each order of the arterial tree  1.38
inlet_rad = 1.8  # inlet radius
order_system = 'strahler'
order_options = 'arterial'
name = 'inlet'
define_rad_from_geom(order_system, s_ratio, name, inlet_rad, order_options, '')
# defines radius by Strahler order in converging (venous mesh)
s_ratio_ven = 1.46  # rate of decrease in radius at each order of the venous tree 1.46
inlet_rad_ven = 4.0  # inlet radius
order_system = 'strahler'
order_options = 'venous'
first_ven_no = ''  # number of elements read in plus one
last_ven_no = ''  # 2x the original number of elements + number of connections
define_rad_from_geom(order_system, s_ratio_ven, first_ven_no, inlet_rad_ven, order_options, last_ven_no)

print('Venous mesh created using parameter and order system:', umbilical_elem_option, order_system)
print('Viscosity:', rheology_type)
print('Vessel type:', vessel_type)

###############################################################
# ---------------- Capillary Creation ----------------------- #
###############################################################

num_convolutes = 10  # number of terminal convolute connections
num_generations = 3  # number of generations of symmetric intermediate villous trees
num_parallel = 6  # number of capillaries per convolute
define_capillary_model(num_convolutes, num_generations, num_parallel, 'byrne_simplified')

#Defining boundary conditions. Value at zero is a dummy variable
if bc_type == 'pressure':
    inlet_pressure = 6650  # Pa (~50mmHg)
    outlet_pressure = 2660  # Pa (~20mmHg)
    inlet_flow = 0  # set to 0 for bc_type = pressure;

if bc_type == 'flow':
    inlet_pressure = 0
    outlet_pressure = 2660
    inlet_flow = 4166.7  # mm3/s

####################################################################################
# ---------------- Solve Pressure, resistance and flow rate----------------------- #
####################################################################################

evaluate_prq(mesh_type, bc_type, rheology_type, vessel_type, inlet_flow, inlet_pressure, outlet_pressure)

print('Pressure and flow simulation complete: ৻(  •̀ ᗜ •́  ৻)')

##export geometry
group_name = 'perf_model'
#Full_flow_tree files include venous mesh that matches arterial mesh
export_1d_elem_geometry(output_flow_dir + 'full_flow_tree_' + sample_number + '.exelem', group_name)
export_node_geometry(output_flow_dir + 'full_flow_tree_' + sample_number + '.exnode', group_name)

# # export element field for radius
field_name = 'radius_perf'
ne_radius = get_ne_radius()
export_1d_elem_field(ne_radius, output_flow_dir + 'radius_perf_' + sample_number + '.exelem', group_name, field_name)
# export flow in each element
field_name = 'flow'
export_1d_elem_field(7, output_flow_dir + 'flow_perf_' + sample_number + '.exelem', group_name, field_name)
#export node field for pressure
field_name = 'pressure_perf'
export_node_field(1, output_flow_dir + 'pressue_perf_' + sample_number + '.exnode', group_name, field_name)
# Export terminal solution
export_terminal_perfusion(output_flow_dir + 'terminal_' + sample_number + '.exnode', 'terminal_soln')
print('Pressure and flow files exported ৻(  •̀ ᗜ •́  ৻)')

####################################################################################
#----------------------------------------------------------------------------------#
#------------------------------- Data extraction ----------------------------------#
#----------------------------------------------------------------------------------#
####################################################################################


###############################################################
# --------------------- Parameters -------------------------- #
###############################################################

#Region of interest : stem_villi or order
ROI = 'order'
#order type (only if order is of importance
order_category = 'strahler'
#Orders of interest
order_interest = [6, 7, 8, 9]
viscosity = 0.0033600

#Needs to have .csv at the end
output_filename = sample_number + '_' + order_category + '.csv'
###############################################################
# ----------------------- File I/O -------------------------- #
###############################################################

nodes_file = pg.import_exnode_tree(output_flow_dir + 'full_flow_tree_' + sample_number + '.exnode')
print('Reading Element file')
elems_file = pg.import_exelem_tree(output_flow_dir + 'full_flow_tree_' + sample_number + '.exelem')
print('Reading Node file')

#nodes_chorion_file = pg.import_exnode_tree(output_tree_dir + 'Umb_' + '.exnode')
#elems_chorion_file = pg.import_exelem_tree(output_tree_dir + 'final_chorion_geom' + '.exelem')
pressure_file = pg.import_exnode_tree(output_flow_dir + 'pressue_perf_' + sample_number + '.exnode')

pressure = pressure_file['nodes']
nodes = full_geom_shaped['nodes']
elems = full_geom_shaped['elems']
nodes_chorion = nodes_Umb
elems_chorion = elems_Umb
print('Reading Radius file')

radii = pg.import_exelem_field(output_flow_dir + 'radius_perf_' + sample_number + '.exelem')

volume, vessel_volumes, lengths = get_vessel_volume(nodes_file['nodes'],radii,elems_file['elems'])
print('total vessel volume: ' + str(volume))
print('Reading Flow file')

flow = pg.import_exelem_field(output_flow_dir + 'flow_perf_' + sample_number + '.exelem')
print('Calculating Orders file')

order_array = pg.evaluate_orders(nodes[:, 1:4], elems)
elements = []
if ROI == 'stem_villi':
    print('Region of Interest: Stem Villi')

    elem_cncty = pg.element_connectivity_1D(nodes_chorion[:, 1:4], elems_chorion)
    elem_up = elem_cncty['elem_up']
    elem_down = elem_cncty['elem_down']
    elem_downstream_end = []
    for i in range(0, len(elem_down)):
        if (elem_down[i, 0] == 0):
            elem_downstream_end.append(i)
    for element in elem_downstream_end:
        shear_stress = (4 * viscosity * flow[element]) / (np.pi * (radii[element] ** 3))
        new_value = np.asarray(
            [elems_chorion[element, 0], radii[element], flow[element], pressure[elems_chorion[element, 1]][1],
             pressure[elems_chorion[element, 2]][1], 0], shear_stress)
        elements.append(new_value)
elif ROI == 'order':
    print('Region of Interest: Order system')

    if order_category == 'strahler':
        order = order_array[order_category]
    elif order_category == 'horsfield':
        order = order_array[order_category]
    elif order_category == 'generation':
        order = order_array[order_category]
    else:
        print('Order category incorrectly defined')
        exit()
    print('Order system: ', order_category)
    interest_elements = []
    max_order = np.max(order)
    #order_interest = [max_order - 1, max_order]
    for elem_i in range(0, len(order)):
        if order[elem_i] in order_interest:
            shear_stress = (4 * viscosity * flow[elem_i]) / (np.pi * (radii[elem_i] ** 3))

            new_value = np.asarray([int(elems[elem_i, 0]), radii[elem_i], flow[elem_i], pressure[elems[elem_i, 1]][1],
                                    pressure[elems[elem_i, 2]][1], shear_stress, order[elem_i]])
            elements.append(new_value)

element_array = np.array(elements)

# Calculate averages for each column (excluding the first column)
average_radius = np.mean(element_array[:, 1])
average_flow_rate = np.mean(element_array[:, 2])
average_inlet_pressure = np.mean(element_array[:, 3])
average_outlet_pressure = np.mean(element_array[:, 4])
median_radius = np.median(element_array[:, 1])
median_flow_rate = np.median(element_array[:, 2])
median_inlet_pressure = np.median(element_array[:, 3])
median_outlet_pressure = np.median(element_array[:, 4])
upper_radius = np.max(element_array[:, 1])
upper_flow_rate = np.max(element_array[:, 2])
upper_inlet_pressure = np.max(element_array[:, 3])
upper_outlet_pressure = np.max(element_array[:, 4])
lower_radius = np.min(element_array[:, 1])
lower_flow_rate = np.min(element_array[:, 2])
lower_inlet_pressure = np.min(element_array[:, 3])
lower_outlet_pressure = np.min(element_array[:, 4])
average_shear_stress = np.mean(element_array[:, 5])
median_shear_stress = np.median(element_array[:, 5])
lower_shear_stress = np.min(element_array[:, 5])
upper_shear_stress = np.max(element_array[:, 5])
print('Average Radius:', average_radius)
print('Average Flow:', average_flow_rate)
print('Average Inlet Pressure:', average_inlet_pressure)
print('Average Outlet Pressure:', average_outlet_pressure)
print('Average Shear Stress: ', average_shear_stress)
# Prepare the average data row

average_row = ['Average', average_radius, average_flow_rate, average_inlet_pressure, average_outlet_pressure, average_shear_stress]
median_row = ['Median', median_radius, median_flow_rate, median_inlet_pressure, median_outlet_pressure, median_shear_stress]
upper_row = ['Maximum', upper_radius, upper_flow_rate, upper_inlet_pressure, upper_outlet_pressure, upper_shear_stress]
lower_row = ['Minimum', lower_radius, lower_flow_rate, lower_inlet_pressure, lower_inlet_pressure, lower_shear_stress]

# Create file if it doesn't exist and write data with headers
try:
    with open(output_flow_dir + output_filename, mode='x', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(
            ['Element Number', 'Radius', 'Flow Rate', 'Inlet Pressure', 'Outlet Pressure', 'Shear Stress', 'Order'])
        # Write data
        writer.writerows(element_array)

        writer.writerow(average_row)
        writer.writerow(median_row)
        writer.writerow(upper_row)
        writer.writerow(lower_row)

    print("File created and data written with headers.")
except FileExistsError:
    with open(output_flow_dir + output_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write data without headers
        writer.writerows(element_array)
        writer.writerow(average_row)
        writer.writerow(median_row)
        writer.writerow(upper_row)
        writer.writerow(lower_row)
    print("File already exists. Appending data without headers.")
