import numpy as np

from included_functions_flow import *
import placentagen as pg
import matplotlib
from matplotlib.pyplot import figure

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from skan import draw, Skeleton, summarize
from fetoflow import *
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
is_rotated = False
constant_vasc_density = True
adjusted_radi = True #Adjusting hte radius of the grown branches
###############################################################
# Parameters that define branching within the placenta volume #
###############################################################/
#Number of seed points targeted for growing tree
n_seed = 32000
weight = 500 #g but is a proxy for cm3 since density of water is 1 g/cm3
Reference_volume = 292062
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
#Fit an ellipse to the placental outline. Weighting to bias the placenta so that more of the
#placental outline is inside the ellipse. This is to find centre point
[x, y, ellipse_fit] = fit_ellipse_2d(placenta_mask, 0.8)
x_mm = ellipse_fit[1] * pixel_scale  #x length of the placenta in mm
y_mm = ellipse_fit[0] * pixel_scale  #y length of the placenta in mm
vol_mm3 = weight * 1000
#volume = 4. * np.pi * x_mm * y_mm * (thickness / 2.) / 3.
thickness = (vol_mm3*6)/(np.pi*y_mm*x_mm*4) #thickness assuming ellipsoid
print(f"X length is {x_mm} and y length is {y_mm}, Calculated Thickness is {thickness}")
#Generate the outline of the placenta in 3D
outputfilename = output_flow_dir + sample_number + '_plac_3d'
plac_outline_nodes = generate_placenta_outline(placenta_mask, pixel_scale, thickness, outputfilename, show_debug_images,
                                               debug_export_all)

#Generate and export nodes that are equally spaces in the 3D spaced placental structure
filename_hull = output_tree_dir + sample_number + '_nodes'
plac_nodes = dict.fromkeys(['nodes'])
plac_nodes['nodes'] = plac_outline_nodes
datapoints, xcentre, ycentre, zcentre, volume = equispaced_data_in_hull(n_seed, plac_nodes)
if debug_export_all:
    pg.export_ex_coords(datapoints, 'placenta', filename_hull, 'exnode')
    print('Node files for placental hull generated and exported to:', filename_hull)





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
if constant_vasc_density:
    n_seed_adjusted = int((volume*n_seed)/Reference_volume)
    print(f"Reference volume is {Reference_volume}, adjusted seed points to {n_seed_adjusted}.")
else:
    n_seed_adjusted = n_seed
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
real_radii = adjust_terminal_branch_radii(nodes_Umb,elems_Umb,real_radii,terminal)
chorion_nodes, chorion_elems, chorion_radii = add_stem_villi(nodes_Umb, elems_Umb, sv_length, terminal, real_radii)
pg.export_exelem_1d(chorion_elems, 'arteries', output_tree_dir + 'chorion')
pg.export_ex_coords(chorion_nodes, 'arteries', output_tree_dir + 'chorion', 'exnode')
pg.export_exfield_1d_linear(chorion_radii, 'placenta', 'radii', output_tree_dir + 'chorion_radii')

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
                                      thickness, 0, ellipse_hull, chorion_and_stem_shaped, 1,parent_list_elems)

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
                                            0, 1.8, 1.53)
outputfilename = output_tree_dir + 'radii_' + sample_number
pg.export_exfield_1d_linear(radii_hull_elem, 'placenta', 'radii', outputfilename)
#ConvertExtoIP(Tree_file)
pg.export_ip_coords(full_geom_shaped['nodes'][:, 1:4], 'placenta', Tree_file)
pg.export_ipelem_1d(full_geom_shaped['elems'], 'placenta', Tree_file)
print('Tree generation complete: ৻(  •̀ ᗜ •́  ৻)')
radii_downstream = set_radii_per_parent(full_geom_shaped,parent_list_nodes,parent_list_elems,chorion_radii,0.06)
pg.export_exfield_1d_linear(radii_downstream, 'placenta', 'radii', output_tree_dir + 'part_radii_' + sample_number)
pg.export_ex_coords(parent_list_nodes, 'placenta', output_tree_dir + 'parent_nodes_' + sample_number, 'exnode')
pg.export_ipfiel(radii_downstream,output_tree_dir + 'tree_radii_' + sample_number)
volume, vessel_volumes, lengths = get_vessel_volume(full_geom_shaped['nodes'],radii_downstream,full_geom_shaped['elems'])
print(f"vessel volume is {vessel_volumes} mm3" )
print(f"volume is {volume}")
pg.calc_terminal_branch(full_geom_shaped['nodes'][:,1:4],full_geom_shaped['elems'])


####################################################################################
#----------------------------------------------------------------------------------#
#------------------------------------ Fetoflow ------------------------------------#
#----------------------------------------------------------------------------------#
####################################################################################

#nodes = read_nodes(img_input_dir + 'full_tree_PN783.ipnode')
#elems = read_elements(img_input_dir + 'full_tree_PN783.ipelem')
#radii = define_fields_from_files({'radius':img_input_dir+'tree_radii_'+sample_number+'.ipfiel'})

nodes = set_nodes_from_array(full_geom_shaped['nodes'])
elems = set_edges_from_array(full_geom_shaped['elems'])
radii = set_fields_from_array(radii_downstream,'radius')
outlet_pressure, inlet_flow, inlet_pressure = 2660, 2083.35, 6650
bcs = generate_boundary_conditions(outlet_pressure=outlet_pressure, inlet_flow=inlet_flow)
# define other required geometric features (radii and decay factors)
umbilical_artery_radius, decay_factor = 1.8, 1.38  # 1.51795
umbilical_vein_radius, decay_factor_vein = 4.0, 1.46
arteries_only = False  # this should rarely be true
viscosity_type = 'constant'  # can also be 'pries_network' or 'pries_vessel' if wanting to incorporate radius-dependence

# Generate the di-graph & calculate the resistances based on the viscosity
print("Creating Geometry")
if adjusted_radi:
    G = create_geometry(nodes, elems, umbilical_artery_radius, decay_factor, umbilical_vein_radius, decay_factor_vein,arteries_only=arteries_only, fields=radii)
else:
    G = create_geometry(nodes, elems, umbilical_artery_radius, decay_factor, umbilical_vein_radius, decay_factor_vein,arteries_only=arteries_only)

print("Adding anastomosis")
G = create_anastomosis(G,2,4,1)
print("Calculating Resistance")
G = calculate_resistance(G, viscosity_model=viscosity_type)
print("Calculating Matrices")
A, b, bc_export = create_small_matrices(G, bcs, branching_angles=False)
print("Solving for Pressures and Flows")
p, q = solve_small_system(A, b, G, bc_export)
G = update_geometry_with_pressures_and_flows(G, p, q)
inlet_measure, outlet_measure = get_tree_properties(G)
#export_region_as_csv(G, 'chorion',output_flow_dir+sample_number+'_ROI.csv', chorion_elems = chorion_elems[:,0], order_interest= None)
print(f"Total vessel volume is {calc_vessel_volume(G,'all')}, arterial vessel volume is {calc_vessel_volume(G, 'artery')}")
#export_all(G, 'placenta', output_flow_dir + 'FF_' + sample_number, 'all')
#export_field(G, 'placenta', 'strahler', output_flow_dir + 'FF_' + sample_number, 'all')
visualise_tree(G, True, 'all')
print('End of Code')
