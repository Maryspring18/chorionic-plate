import pandas as pd
from fetoflow import *
import placentagen as pg
import numpy as np
import matplotlib.image as mpimg
from skimage import filters, measure, color
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import Delaunay
from scipy import spatial as sp
import os
def main():
    ###############################################################
    # Parameters that define branching within the placenta volume #
    ###############################################################/
    # Number of seed points targeted for growing tree
    n_seed = 32000
    # placenta measurements
    thickness = 20  # mm

    Grow_tree_root  = "W:/intermediate/2023-sex-specific/chorionic segmentations/outputs_grow_tree/"
    Sample_input_root = "W:/derivative/2023-sex-specific/chorionic segmentations/"
    Output_dir = "outputs/" #TODO
    csv_file  = "Placenta_codes.csv" #TODO

    df = pd.read_csv(csv_file)
    #sample_list = df['Code'].tolist()
    sample_list = df.to_numpy().tolist()
    out_list = []

    for sample,group in sample_list:
        print('Current sample: '+sample + ', Group: ' + group)
        sample_input_dir = Sample_input_root + sample #TODO
        grow_tree_dir = Grow_tree_root  + sample #TODO

        # ARTERIAL VESSEL VOLUME
        print("Generating vessel volume")
        node_file_path = grow_tree_dir + '/full_tree_' + sample + '.exnode'
        elem_file_path = grow_tree_dir + '/full_tree_' + sample + '.exelem'
        radii_file_path = grow_tree_dir + '/radii_' + sample + '.exelem'
        if os.path.isfile(node_file_path) and os.path.isfile(elem_file_path) and os.path.isfile(radii_file_path):
            nodes =  read_nodes_exnode(node_file_path)
            elems = read_edges_exelem(elem_file_path)
            fields = define_fields_from_exelem(radii_file_path, 'radii')
        else:
            print("cannot find files, Moving onto next sample")
            continue

        print('Creating Geometry')
        G = create_geometry(nodes, elems, 1.8, 1.36,4, 1.56, arteries_only=True,fields=fields)
        volume_vessels = 0
        for u,v,data in G.edges(data=True):
            length = calcLength(G,u,v)
            radius = data['radius']
            volume_edge = np.pi * (radius**2) * length
            volume_vessels += volume_edge
        print('Total vessel volume: ' + str(volume_vessels))
        #######################################################################
        # -------------------Ellipse/Hull Generation---------------------------#
        #######################################################################

        #Get scale
        scale_filename = Sample_input_root + sample +  '/'+ sample+ '_scale.png'
        placenta_filename = Sample_input_root + sample + '/'+ sample+ '_outline.png'
        if (os.path.isfile(scale_filename) == False) and (os.path.isfile(scale_filename) == False):
            continue
        scale_file = read_png(scale_filename, 'g')
        pixel_scale = get_scale(10, scale_file)
        pixel_scale = pixel_scale * 1.
        print('pixel_scale: ' + str(pixel_scale))
        # read placenta outline

        placenta_mask = read_png(placenta_filename, 'g') #TODO
        print(placenta_mask.shape)
        # Generate the outline of the placenta in 3D
        outputfilename = Output_dir + '_plac_3d'

        plac_outline_nodes = generate_placenta_outline(placenta_mask, pixel_scale, thickness, outputfilename,
                                                       False,
                                                       False, False, 10)

        # Generate and export nodes that are equally spaces in the 3D spaced placental structure
        filename_hull = Output_dir + '_nodes'
        plac_nodes = dict.fromkeys(['nodes'])
        plac_nodes['nodes'] = plac_outline_nodes
        datapoints, xcentre, ycentre, zcentre, volume = equispaced_data_in_hull(n_seed, plac_nodes, False)
        # Fit an ellipse to the placental outline. Weighting to bias the placenta so that more of the
        # placental outline is inside the ellipse. This is to find centre point
        [x, y, ellipse_fit] = fit_ellipse_2d(placenta_mask, 0.8)
        x_mm = ellipse_fit[1] * pixel_scale  # x length of the placenta in mm
        y_mm = ellipse_fit[0] * pixel_scale  # y length of the placenta in mm
        volume = 4. * np.pi * x_mm * y_mm * (thickness / 2.) / 3.

        # -------------------Transform the 3D hull ---------------------------#
        # Calculate the desired center in real-world coordinates
        desired_center = np.array([ellipse_fit[3] * pixel_scale, ellipse_fit[4] * pixel_scale, zcentre])

        # Translate the 3D points. This is the real world coordinates used for generation
        translated_points_3d = datapoints - desired_center
        datapoints_ellipse, hull_params = generate_ellipse_hull(translated_points_3d)
        # datapoints_ellipse = translated_points_3d
        datapoints_ellipse_array = np.array(datapoints_ellipse)

        index = np.arange(datapoints_ellipse_array.shape[0]).reshape(-1, 1)  # 0-based indexing

        datapoints_ellipse_array = np.hstack((index, datapoints_ellipse_array))
        plac_nodes = dict.fromkeys(['nodes'])
        plac_nodes['nodes'] = datapoints_ellipse_array

        ellipse_hull, xcentre, ycentre, zcentre, volume_hull = equispaced_data_in_hull(n_seed, plac_nodes, True)
        print('Total placental volume: ' + str(volume_hull))
        out_list.append((sample,group,volume_vessels,volume_hull))
    header = ['Code','Group','Vessel Volume', 'Placenta Volume']
    df = pd.DataFrame(out_list, columns=header)
    df.to_csv(Output_dir + 'output_data.csv', index=False)



def read_png(filename, extract_colour):
    #This function reads in a png file and extract the relevant colour from the image
    img1 = mpimg.imread(filename)
    if extract_colour == 'all':
        img2 = img1
    elif extract_colour == 'r':
        img2 = img1[:, :, 0]
    elif extract_colour == 'g':
        img2 = img1[:, :, 1]
    elif extract_colour == 'b':
        img2 = img1[:, :, 2]
    else:  #default to all channels
        img2 = img1
    return img2


def generate_placenta_outline(image, pixel_spacing, thickness, outputfilename, debug_img, debug_file, is_rotate, rotation_angle):

    edges = filters.sobel(image)
    binary_edges = edges > filters.threshold_otsu(edges)
    contours = measure.find_contours(binary_edges, level=0.8)

    # Assume the largest contour is the desired shape
    largest_contour = max(contours, key=len)

    # Plot the contour on the original image

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(largest_contour[:, 1], largest_contour[:, 0], linewidth=2, color='red')
    ax.set_title('Detected Contour')
    if debug_img:
        plt.show()

    # Extract contour points
    contour_points_mm = [(x * pixel_spacing, y * pixel_spacing) for y, x in largest_contour]

    nodes = np.zeros((len(contour_points_mm) * 3, 4))
    node_count = 0
    for point in contour_points_mm:
        for dim in range(0, 3):
            nodes[node_count, 0] = node_count  #node number
            nodes[node_count, 1] = point[1]  # X coordinate
            nodes[node_count, 2] = point[0]  # Y coordinate
            if dim == 0:
                nodes[node_count, 3] = 0.0  # Y coordinate
            elif dim == 1:
                nodes[node_count, 3] = thickness / 2.0
            elif dim == 2:
                nodes[node_count, 3] = thickness
            node_count += 1
    if debug_file:
        pg.export_ex_coords(nodes[:, :][0:node_count + 1], 'placenta_3d', outputfilename, 'exnode')
        print('Exported Outline Data to: ', outputfilename)
    print('3D Placenta outline generation complete. File output: ', debug_file)
    return nodes[:, :][0:node_count + 1]


def fit_ellipse_2d(img, weight):
    #x coordinates of placental edge
    surface_points_x = np.nonzero(img)[0].astype(float)  # np.zeros([np.count_nonzero(img),2])

    #y coordinates of placental edge
    surface_points_y = np.nonzero(img)[1].astype(float)

    com_start = [np.mean(surface_points_x), np.mean(surface_points_y)]
    #rough centre point X
    x_radius_start = (np.max(surface_points_x) - np.min(surface_points_x)) / 2.
    #rough centre coordinate Y
    y_radius_start = (np.max(surface_points_y) - np.min(surface_points_y)) / 2.
    alpha_start = 0.
    #optimizes using least squared an ellipse that fits the outline of the placenta
    opt = least_squares(distance_from_ellipse,
                        [x_radius_start, y_radius_start, alpha_start, com_start[0], com_start[1]],
                        args=(surface_points_x, surface_points_y, weight), xtol=1e-8, verbose=0)

    return surface_points_x, surface_points_y, opt.x


def distance_from_ellipse(params, surface_x, surface_y, penalisation_factor):
    x_rad = params[0]
    y_rad = params[1]
    alpha = params[2]
    #offset surface to current COM
    surface_x = surface_x - params[3]
    surface_y = surface_y - params[4]

    A = ((np.cos(alpha) / x_rad) ** 2. + (np.sin(alpha) / y_rad) ** 2.) * np.multiply(surface_x, surface_x)
    B = 2.0 * np.cos(alpha) * np.sin(alpha) * (1. / x_rad ** 2. - 1. / y_rad ** 2.) * np.multiply(surface_x, surface_y)
    C = (np.sin(alpha) / x_rad) ** 2. + (np.cos(alpha) / y_rad) ** 2. * np.multiply(surface_x, surface_x)
    distance = A + B + C - 1.
    if (x_rad > np.max(abs(surface_x)) and y_rad > np.max(abs(surface_y))):
        distance = distance
    else:  #penalise the ellipsoid being inside the structure
        distance = distance * penalisation_factor

    distance = np.sum(distance ** 2)
    return distance


def generate_ellipse_hull(datapoints):
    mean = np.mean(datapoints[:, 2])
    max_x = 0
    min_x = 0

    index = 0
    thickness = max(datapoints[:, 2]) - min(datapoints[:, 2])
    rz = thickness / 2.0
    filtered_y_list = np.unique(datapoints[:, 1])
    maxmin_x = np.zeros((len(filtered_y_list), 5))
    #find bounds of x for each y slice
    for y_index in filtered_y_list:
        for point in datapoints:
            if point[1] == y_index:
                if point[0] < min_x:
                    min_x = point[0]
                if point[0] > max_x:
                    max_x = point[0]

                maxmin_x[index, 0] = y_index
                maxmin_x[index, 1] = min_x
                maxmin_x[index, 2] = max_x
        index += 1
        min_x = 0
        max_x = 0
    #Generate array of rx and a (offset) for each y slice

    for i in range(0, len(maxmin_x)):
        rx = (maxmin_x[i, 2] - maxmin_x[i, 1]) / 2.0
        a = (maxmin_x[i, 2] + maxmin_x[i, 1]) / 2.0
        maxmin_x[i, 3] = rx + 2
        maxmin_x[i, 4] = a
    datapoints_ellipse = []
    for point in datapoints:
        indices = np.where(maxmin_x[:, 0] == point[1])[0]
        coord_check = check_in_ellipse(point[0], point[2], maxmin_x[indices, 3], maxmin_x[indices, 4], rz)
        if coord_check == True:
            datapoints_ellipse.append(point)

    return datapoints_ellipse, maxmin_x


def check_in_ellipse(x, z, rx, a, rz):
    in_ellipse = False
    check_value = (((x - a) / rx) ** 2) + ((z / rz) ** 2)
    if check_value < 1.0:
        in_ellipse = True
    return in_ellipse

def get_scale(scalebar_size, image_array):

    #binary = img > 1.0e-6  #all non zeros

    line_pixels = np.where(image_array > 0.5)

    if len(line_pixels[0]) == 0:
        raise ValueError("No line detected in the image")

    # Extract the x-coordinates of the line
    x_coords = line_pixels[1]
    y_coords = line_pixels[0]
    # Calculate the length of the line in pixels
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)

    if x_range > y_range:  # Horizontal line
        length_in_pixels = x_range + 1
        print("Scale bar horizontal")
    else:  # Vertical line
        length_in_pixels = y_range + 1
        print("Scale bar vertical")

    print('Length of bar in pixels: ', length_in_pixels)


    # Calculate scale in mm/pixel
    scale_mm_per_pixel = scalebar_size / length_in_pixels
    scale_mm_per_pixel = np.round(scale_mm_per_pixel,4)

    return scale_mm_per_pixel


def equispaced_data_in_hull(n, geom, volume_out):
    hull = sp.ConvexHull(geom['nodes'][:, 1:4])
    # for i in range(0, len(hull.vertices)):
    xmin = np.min(geom['nodes'][hull.vertices, 1])
    xmax = np.max(geom['nodes'][hull.vertices, 1])
    ymin = np.min(geom['nodes'][hull.vertices, 2])
    ymax = np.max(geom['nodes'][hull.vertices, 2])
    zmin = np.min(geom['nodes'][hull.vertices, 3])
    zmax = np.max(geom['nodes'][hull.vertices, 3])
    xcentre = (xmax - xmin) / 2
    ycentre = (ymax - ymin) / 2
    zcentre = (zmax - zmin) / 2

    cuboid_vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)  # cuboid volume
    total_n = (cuboid_vol / hull.volume) * n
    data_spacing = (cuboid_vol / total_n) ** (1.0 / 3.0)

    nd_x = np.floor((xmax - xmin) / data_spacing)
    nd_y = np.floor((ymax - ymin) / data_spacing)
    nd_z = np.floor((zmax - zmin) / data_spacing)
    nd_x = int(nd_x)
    nd_y = int(nd_y)
    nd_z = int(nd_z)

    # Set up edge node coordinates
    x_coord = np.linspace(xmin, xmax, nd_x)
    y_coord = np.linspace(ymin, ymax, nd_y)
    z_coord = np.linspace(zmin, zmax, nd_z)
    # p = 1.5  # Adjust power: p > 1 → more points near zmin
    # z_coord = zmin + (zmax - zmin) * (np.linspace(0, 1, nd_z) ** p)

    # Use these vectors to form a unifromly spaced grid
    data_coords = np.vstack(np.meshgrid(x_coord, y_coord, z_coord)).reshape(3, -1).T

    # Store nodes that lie within hull
    num_data = 0  # zero the total number of data points
    datapoints = np.zeros((nd_x * nd_y * nd_z, 3))
    hull2 = sp.Delaunay(geom['nodes'][hull.vertices, 1:4])
    for i in range(len(data_coords)):  # Loop through grid
        if hull2.find_simplex(data_coords[i]) > 0:
            coord_check = True
        else:
            coord_check = False
        if coord_check is True:  # Has to be strictly in the hull
            datapoints[num_data, :] = data_coords[i, :]  # add to data array
            num_data = num_data + 1
    datapoints.resize(num_data, 3, refcheck=False)  # resize data array to correct size
    # volume calculation
    total_vol = 0
    if volume_out:
        triangulation = sp.Delaunay(datapoints)
        simplices = datapoints[triangulation.simplices]
        volumes = np.abs(np.linalg.det(simplices[:, 1:] - simplices[:, 0, np.newaxis]) / 6)
        total_vol = np.sum(volumes)
        print('Data points within hull allocated. Total = ' + str(len(datapoints)))
        print('Total volume  = ' + str(total_vol))
    return datapoints, xcentre, ycentre, zcentre, total_vol


if __name__ == '__main__':
    main()