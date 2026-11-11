from fetoflow import *
import csv
import os
from included_functions_flow import *
root = '/home/cjay511/Documents/Outputs'
sample_number = 'P49_uct'
img_input_dir = root+ '/input/' +sample_number +'/'
output_tree_dir = root+ '/outputs/' + sample_number + '/outputs_grow_tree/'
output_flow_dir = root+ '/outputs/' + sample_number + '/outputs_flow_tree/base/base/'
output_table_dir = root+ '/outputs/' + sample_number + '/outputs_branch/'
#img_input_dir = 'W:/derivative/2023-sex-specific/chorionic-segmentations/' +sample_number +'/'
#output_tree_dir = 'W:/intermediate/2023-sex-specific/chorionic-segmentations/' + sample_number + '/outputs_grow_tree/'
#output_flow_dir = 'W:/intermediate/2023-sex-specific/chorionic-segmentations/' + sample_number + '/outputs_flow_tree/'
#output_table_dir = 'W:/intermediate/2023-sex-specific/chorionic-segmentations/' + sample_number + '/outputs_branch/'
#img_input_dir = '/media/share/intermediate/2023-sex-specific/chorionic-segmentations/' +sample_number +'/'
if not os.path.exists(output_flow_dir):
    os.makedirs(output_flow_dir)
####################################################################################
#----------------------------------------------------------------------------------#
#------------------------------------ Fetoflow ------------------------------------#
#----------------------------------------------------------------------------------#
####################################################################################


#nodes = read_nodes(output_tree_dir + 'full_tree_' +sample_number+ '.ipnode')
#elems = read_elements(output_tree_dir + 'full_tree_' + sample_number+ '.ipelem')
#radii = define_fields_from_files({'radius':img_input_dir+'tree_radii_'+sample_number+'.ipfiel'})


nodes = read_nodes_exnode(output_tree_dir + 'full_tree_' +sample_number+ '.exnode')
elems = read_edges_exelem(output_tree_dir + 'full_tree_' + sample_number+ '.exelem')
radii = define_fields_from_exelem(output_tree_dir+'tree_radii_'+sample_number+'.exelem', 'radius')

#nodes = set_nodes_from_array(full_geom_shaped['nodes'])
#elems = set_edges_from_array(full_geom_shaped['elems'])
#radii = set_fields_from_array(radii_downstream,'radius')
outlet_pressure, inlet_flow, inlet_pressure = 2660, 2083.35, 6650
bcs = generate_boundary_conditions(outlet_pressure=outlet_pressure, inlet_flow=inlet_flow)
# define other required geometric features (radii and decay factors)
umbilical_artery_radius, decay_factor = 1.8, 1.43 # 1.51795
umbilical_vein_radius, decay_factor_vein = 4.0, 1.46
arteries_only = False  # this should rarely be true
viscosity_type = 'constant'  # can also be 'pries_network' or 'pries_vessel' if wanting to incorporate radius-dependence

# Generate the di-graph & calculate the resistances based on the viscosity
print("Creating Geometry")
G = create_geometry(nodes, elems, umbilical_artery_radius, decay_factor, umbilical_vein_radius, decay_factor_vein,arteries_only=arteries_only, fields=radii)


print("Adding anastomosis")
#G = create_anastomosis(G,140,1355,1) #P51
G = create_anastomosis(G,1831,2033,1)
print("Adding occlusion")
#G = update_edge_property(G,3158,'radius',0.01)
print("Calculating Resistance")
G = calculate_resistance(G, viscosity_model=viscosity_type)
H = G.copy()
print("Calculating Matrices")
A, b, bc_export = create_small_matrices(G, bcs, branching_angles=False)
print("Solving for Pressures and Flows")
p, q = solve_small_system(A, b, G, bc_export)
G = update_geometry_with_pressures_and_flows(G, p, q)
inlet_measure, outlet_measure = get_tree_properties(G)
#export_region_as_csv(G, 'chorion',output_flow_dir+sample_number+'_ROI.csv', chorion_elems = chorion_elems[:,0], order_interest= None)
print(f"Total vessel volume is {calc_vessel_volume(G,'all')}, arterial vessel volume is {calc_vessel_volume(G, 'artery')}")
export_all(G, 'placenta', output_flow_dir + 'uct_' + sample_number, 'arteries')
#export_field(G, 'placenta', 'strahler', output_flow_dir + 'FF_' + sample_number, 'all')
A, df = extract_terminal_vessel_data(G)
R = arterial_tree_resistances(G)
print(R)
print(f"Number of terminal vessels is {df.shape[0]}")
count = 0
for data in df['flow']:
    if data < 0.001:
        count += 1
print(f"There are {count} vessel with less than 0.01 mm3/s flow")
df.to_csv(output_flow_dir+'terminals_'+sample_number+'.csv',index=False)
print("Calculating whole tree resistance using boundary conditions")
bcs2 = generate_boundary_conditions(inlet_pressure=inlet_pressure, outlet_pressure=outlet_pressure)
C,d,bc_export2 = create_small_matrices(H,bcs2,branching_angles=False)
p2,q2 = solve_small_system(C,d,H,bc_export2)
H = update_geometry_with_pressures_and_flows(G,p2,q2)
inlet_measure2, outlet_measure2 = get_tree_properties(H)
tree_resistance = (inlet_pressure-outlet_pressure)/next(iter(outlet_measure2.values()))['flow']
print(f"Whole tree resistance is {tree_resistance}")
visualise_tree(G, True, 'arteries')
print('End of Code')
