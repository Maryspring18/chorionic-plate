from fetoflow import *
import argparse
import os
from fetoflow.solve_utils import solve_iterative_system
from networkx.classes import number_of_nodes, number_of_edges

parser = argparse.ArgumentParser()
parser.add_argument("--sample")
args = parser.parse_args()
sample_number = args.sample

use_real_radii = True
output_tree_dir = 'X:/intermediate/2023-sex-specific/chorionic-segmentations/' + sample_number + '/outputs_grow_tree/'
output_flow_dir = 'X:/intermediate/2023-sex-specific/chorionic-segmentations/passive_pries_noanast/' + sample_number + '/outputs_flow_tree/'
if not os.path.exists(output_flow_dir):
    os.makedirs(output_flow_dir)

nodes = read_nodes(output_tree_dir + 'full_tree_' +sample_number+ '.ipnode')
elems = read_elements(output_tree_dir + 'full_tree_' + sample_number+ '.ipelem')
radii = define_fields_from_files({'radius':output_tree_dir+'tree_radii_'+sample_number+'.ipfiel'})

#nodes = set_nodes_from_array(full_geom_shaped['nodes'])
#elems = set_edges_from_array(full_geom_shaped['elems'])
#radii = set_fields_from_array(radii_downstream,'radius')
outlet_pressure, inlet_flow, inlet_pressure = 2660, 2083.35, 6650
bcs = generate_boundary_conditions(outlet_pressure=outlet_pressure, inlet_flow=inlet_flow)
# define other required geometric features (radii and decay factors)
umbilical_artery_radius, decay_factor = 1.8, 1.38  # 1.51795
umbilical_vein_radius, decay_factor_vein = 4.0, 1.46
arteries_only = False  # this should rarely be true
viscosity_type = 'constant'  # can also be 'pries_network' or 'pries_vessel' if wanting to incorporate radius-dependence
vessel_elasticity = 'rigid'
# Generate the di-graph & calculate the resistances based on the viscosity
print("Creating Geometry")
if use_real_radii:
    G = create_geometry(nodes, elems, umbilical_artery_radius, decay_factor, umbilical_vein_radius, decay_factor_vein,arteries_only=arteries_only, fields=radii)
else:
    G = create_geometry(nodes, elems, umbilical_artery_radius, decay_factor, umbilical_vein_radius, decay_factor_vein,
                        arteries_only=arteries_only)
#print("Adding anastomosis")
#G = create_anastomosis(G,2,4,1)
print("Calculating Resistance")
G = calculate_resistance(G, viscosity_model=viscosity_type)
if viscosity_type == 'constant' and vessel_elasticity == 'rigid':
    print("Calculating Matrices")
    A, b, bc_export = create_small_matrices(G, bcs, branching_angles=False)
    print("Solving for Pressures and Flows")
    p, q = solve_small_system(A, b, G, bc_export)
else:
    print("Calculating Matrices")
    A,b = create_matrices(G,bcs)
    print("Iteratively Solving for Pressures and Flows")
    p, q = solve_iterative_system(G,A,b,num_nodes=number_of_nodes(G), num_edges=number_of_edges(G),bcs=bcs,viscosity_model=viscosity_type,mu=0.33600e-02,capillary_model='analytical2015', capillary_parameters=None, vessel_elasticity='passive')

G = update_geometry_with_pressures_and_flows(G, p, q)
inlet_measure, outlet_measure = get_tree_properties(G)
print(f"Total vessel volume is {calc_vessel_volume(G,'all')}, arterial vessel volume is {calc_vessel_volume(G, 'artery')}")
export_all(G, 'placenta', output_flow_dir  + sample_number, 'all')
export_field(G, 'placenta', 'strahler', output_flow_dir + 'FF_' + sample_number, 'all')
#visualise_tree(G, True, 'all')
print('End of Code')
