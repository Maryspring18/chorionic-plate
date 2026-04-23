import numpy as np
import placentagen as pg
import matplotlib.image as mpimg
from skimage import filters, measure, color
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt
from scipy import spatial as sp
from skimage.morphology import skeletonize  # Compute the skeleton of a binary image
from skan import csr
from skan import Skeleton, summarize
import networkx as nx
import cv2
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"



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


def calculate_area(image_path, scale, is_debug_image):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green color in HSV
    lower_green = np.array([35, 50, 50])  # Adjust as needed
    upper_green = np.array([85, 255, 255])

    # Create a binary mask where green is detected
    mask = cv2.inRange(hsv, lower_green, upper_green)
    if is_debug_image:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Green Mask")
        plt.axis("off")

        plt.show()
    # Count the number of nonzero pixels (green area)
    green_pixels = np.count_nonzero(mask)

    # Convert pixels to real-world area (scale is in mm/pixel)
    area_mm2 = green_pixels * (scale ** 2)

    return green_pixels, area_mm2





def read_png(filename, extract_colour):
    # This function reads in a png file and extract the relevant colour from the image
    img1 = mpimg.imread(filename)
    if extract_colour == 'all':
        img2 = img1
    elif extract_colour == 'r':
        img2 = img1[:, :, 0]
    elif extract_colour == 'g':
        img2 = img1[:, :, 1]
    elif extract_colour == 'b':
        img2 = img1[:, :, 2]
    else:  # default to all channels
        img2 = img1
    return img2


def generate_placenta_outline(image, pixel_spacing, thickness, outputfilename, debug_img, debug_file,
                              ):

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
    #   if debug_img:
    # plt.show()

    # Extract contour points
    contour_points_mm = [(x * pixel_spacing, y * pixel_spacing) for y, x in largest_contour]

    nodes = np.zeros((len(contour_points_mm) * 3, 4))
    node_count = 0
    for point in contour_points_mm:
        for dim in range(0, 3):
            nodes[node_count, 0] = node_count  # node number
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
    # x coordinates of placental edge
    surface_points_x = np.nonzero(img)[0].astype(float)  # np.zeros([np.count_nonzero(img),2])

    # y coordinates of placental edge
    surface_points_y = np.nonzero(img)[1].astype(float)

    com_start = [np.mean(surface_points_x), np.mean(surface_points_y)]
    # rough centre point X
    x_radius_start = (np.max(surface_points_x) - np.min(surface_points_x)) / 2.
    # rough centre coordinate Y
    y_radius_start = (np.max(surface_points_y) - np.min(surface_points_y)) / 2.
    alpha_start = 0.
    # optimizes using least squared an ellipse that fits the outline of the placenta
    opt = least_squares(distance_from_ellipse,
                        [x_radius_start, y_radius_start, alpha_start, com_start[0], com_start[1]],
                        args=(surface_points_x, surface_points_y, weight), xtol=1e-8, verbose=0)

    return surface_points_x, surface_points_y, opt.x


def distance_from_ellipse(params, surface_x, surface_y, penalisation_factor):
    x_rad = params[0]
    y_rad = params[1]
    alpha = params[2]
    # offset surface to current COM
    surface_x = surface_x - params[3]
    surface_y = surface_y - params[4]

    A = ((np.cos(alpha) / x_rad) ** 2. + (np.sin(alpha) / y_rad) ** 2.) * np.multiply(surface_x, surface_x)
    B = 2.0 * np.cos(alpha) * np.sin(alpha) * (1. / x_rad ** 2. - 1. / y_rad ** 2.) * np.multiply(surface_x, surface_y)
    C = (np.sin(alpha) / x_rad) ** 2. + (np.cos(alpha) / y_rad) ** 2. * np.multiply(surface_x, surface_x)
    distance = A + B + C - 1.
    if (x_rad > np.max(abs(surface_x)) and y_rad > np.max(abs(surface_y))):
        distance = distance
    else:  # penalise the ellipsoid being inside the structure
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
    # find bounds of x for each y slice
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
    # Generate array of rx and a (offset) for each y slice

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


def equispaced_data_in_hull(n, geom):
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
    triangulation = sp.Delaunay(datapoints)
    simplices = datapoints[triangulation.simplices]
    volumes = np.abs(np.linalg.det(simplices[:, 1:] - simplices[:, 0, np.newaxis]) / 6)
    total_vol = np.sum(volumes)
    print('Data points within hull allocated. Total = ' + str(len(datapoints)))
    print('Total volume  = ' + str(total_vol))
    return datapoints, xcentre, ycentre, zcentre, total_vol

def skeletonise_2d(img):
    # convert img to binary
    binary = img > 1.0e-6  # all non zeros
    sk = skeletonize(binary, method='zhang')  # skeletonize binary image
    return sk


def new_branch(mydegrees, branch_data, coordinates, pixel_graph, i, node_kount, elem_kount, nodes, elems, node_map):
    parent_list = np.zeros(3)
    # print('parent node in',i, mydegrees[i])
    # i is the 'old' node number, from the skeletonisation, indexed ftom 1

    continuing = True
    xcord, ycord = coordinates

    currentdegrees = 2  # dummy to enter loop
    while currentdegrees == 2:  # while a continuation branch
        count_new_vessel = 0  # not a new vessel
        for j in range(pixel_graph.indptr[i], pixel_graph.indptr[i + 1]):  # looking at all branches connected to inew
            inew = pixel_graph.indices[j]  # index of connected branch (old) indexed from one
            np_old = np.where(node_map == i)  # node number
            # need to find the index of
            if inew not in node_map:
                currentdegrees = mydegrees[inew]  # how many branches this node is connected to
                count_new_vessel = count_new_vessel + 1  # create a new vessel segment
                # Create new node
                node_kount = node_kount + 1  # create a new node
                node_map[node_kount] = inew  # Mapping old to new node number
                nodes[node_kount, 0] = node_kount  # new node number
                nodes[node_kount, 1] = xcord[inew]  # coordinates indexed to 'old' node number
                nodes[node_kount, 2] = ycord[inew]
                # plt.plot(coordinates[inew,0],coordinates[inew,1],'+')
                nodes[node_kount, 3] = 0.0  # dummy z-coord
                # Create new element
                elem_kount = elem_kount + 1
                elems[elem_kount, 0] = elem_kount
                elems[elem_kount, 1] = np_old[0]
                elems[elem_kount, 2] = node_kount
                iold = i
                i = inew
            if j == (pixel_graph.indptr[
                         i + 1] - 1) and count_new_vessel == 0:  # a get out which basically throws out loops
                mydegrees[i] = 10
                currentdegrees = 10

    if currentdegrees == 1:  # Terminal
        continuing = False
    elif currentdegrees == 3:  # bifurcation
        loops = False
        # need to check if isolated loop
        k = np.where(np.asarray(branch_data['node-id-src']) == i)
        if k[0].size >= 1:
            for i_branch in range(0, k[0].size):
                if branch_data['branch-type'][k[0][i_branch]] == 3:
                    loops = True
        if not loops:
            pl_kount = 0
            node_kount_bif = node_kount
            for j in range(pixel_graph.indptr[i], pixel_graph.indptr[i + 1]):
                inew = pixel_graph.indices[j]
                if inew not in node_map:
                    node_kount = node_kount + 1
                    node_map[node_kount] = inew
                    nodes[node_kount, 0] = node_kount
                    nodes[node_kount, 1] = xcord[inew]
                    nodes[node_kount, 2] = ycord[inew]
                    nodes[node_kount, 3] = 0.0  # dummy z-coord
                    elem_kount = elem_kount + 1
                    elems[elem_kount, 0] = elem_kount
                    elems[elem_kount, 1] = node_kount_bif
                    elems[elem_kount, 2] = node_kount
                    parent_list[pl_kount] = inew
                    pl_kount = pl_kount + 1
            continuing = True
        else:
            continuing = False
    else:  # for now exclude morefications and treat as a terminal
        continuing = False
        if mydegrees[i] == 4 or mydegrees[i] == 5:
            i_trif = i
            node_map_temp = np.copy(node_map)
            node_kount_temp = node_kount
            parent_list_trif = np.zeros(4, dtype=int)
            true_branches = np.zeros(4, dtype=int)
            true_trifurcation = True
            pl_kount = 0
            true_branches_kount = 0
            for j in range(pixel_graph.indptr[i], pixel_graph.indptr[i + 1]):
                inew = pixel_graph.indices[j]
                if inew not in node_map_temp:
                    if mydegrees[inew] != 2:
                        true_trifurcation = False
                    else:
                        parent_list_trif[pl_kount] = inew
                        pl_kount = pl_kount + 1
            if true_trifurcation:
                for br in parent_list_trif:
                    br_length = 0.0
                    i = br
                    currentdegrees = mydegrees[i]
                    while currentdegrees == 2:  # while a continuation branch
                        count_new_vessel = 0  # not a new vessel
                        for j in range(pixel_graph.indptr[i],
                                       pixel_graph.indptr[i + 1]):  # looking at all branches connected to inew
                            inew = pixel_graph.indices[j]  # index of connected branch (old) indexed from one
                            # need to find the index of
                            if inew not in node_map_temp:
                                currentdegrees = mydegrees[inew]  # how many branches this node is connected to
                                count_new_vessel = count_new_vessel + 1  # create a new vessel segment
                                br_length = br_length + 1
                                node_kount_temp = node_kount_temp + 1  # create a new node
                                node_map_temp[node_kount_temp] = inew  # Mapping old to new node number
                                i = inew
                            if j == (pixel_graph.indptr[
                                         i + 1] - 1) and count_new_vessel == 0:  # a get out which basically throws out loops
                                currentdegrees = 10
                    if br_length > 0.0:
                        # True branches
                        true_branches[true_branches_kount] = br
                        true_branches_kount = true_branches_kount + 1
            if true_branches_kount == 0:
                continuing = False
            elif true_branches_kount == 1:
                # This is a single branch that should continue from here
                inew = true_branches[0]
                pl_kount = 0
                node_kount_bif = node_kount
                node_kount = node_kount + 1
                node_map[node_kount] = inew
                nodes[node_kount, 0] = node_kount
                nodes[node_kount, 1] = xcord[inew]
                nodes[node_kount, 2] = ycord[inew]
                nodes[node_kount, 3] = 0.0  # dummy z-coord
                elem_kount = elem_kount + 1
                elems[elem_kount, 0] = elem_kount
                elems[elem_kount, 1] = node_kount_bif
                elems[elem_kount, 2] = node_kount
                parent_list[pl_kount] = inew
                pl_kount = pl_kount + 1
                continuing = True
            elif true_branches_kount == 2:
                # This is a single branch that should continue from here
                pl_kount = 0
                node_kount_bif = node_kount
                for br in range(0, 2):
                    inew = true_branches[br]
                    node_kount = node_kount + 1
                    node_map[node_kount] = inew
                    nodes[node_kount, 0] = node_kount
                    nodes[node_kount, 1] = xcord[inew]
                    nodes[node_kount, 2] = ycord[inew]
                    nodes[node_kount, 3] = 0.0  # dummy z-coord
                    elem_kount = elem_kount + 1
                    elems[elem_kount, 0] = elem_kount
                    elems[elem_kount, 1] = node_kount_bif
                    elems[elem_kount, 2] = node_kount
                    parent_list[pl_kount] = inew
                    pl_kount = pl_kount + 1
                continuing = True
            else:
                continuing = True
                # TRUE TRIFURCATION
                pl_kount = 0
                node_kount_bif = node_kount
                # create a branch to the first point
                inew = true_branches[0]
                node_kount = node_kount + 1
                node_map[node_kount] = inew
                nodes[node_kount, 0] = node_kount
                nodes[node_kount, 1] = xcord[inew]
                nodes[node_kount, 2] = ycord[inew]
                nodes[node_kount, 3] = 0.0  # dummy z-coord
                elem_kount = elem_kount + 1
                elems[elem_kount, 0] = elem_kount
                elems[elem_kount, 1] = node_kount_bif
                elems[elem_kount, 2] = node_kount
                parent_list[pl_kount] = inew
                pl_kount = pl_kount + 1
                # Create a second branch to the ave coord of the parent and the other two branches
                x_coord = nodes[node_kount_bif, 1]
                y_coord = nodes[node_kount_bif, 2]
                for br in range(1, 3):
                    x_coord = x_coord + xcord[true_branches[br]]
                    y_coord = y_coord + ycord[true_branches[br]]
                x_coord = x_coord / 3.
                y_coord = y_coord / 3.
                node_kount = node_kount + 1
                nodes[node_kount, 0] = node_kount
                nodes[node_kount, 1] = x_coord
                nodes[node_kount, 2] = y_coord
                nodes[node_kount, 3] = 0.0  # dummy z-coord
                elem_kount = elem_kount + 1
                elems[elem_kount, 0] = elem_kount
                elems[elem_kount, 1] = node_kount_bif
                elems[elem_kount, 2] = node_kount
                node_kount_bif = node_kount
                for br in range(1, 3):
                    inew = true_branches[br]
                    node_kount = node_kount + 1
                    node_map[node_kount] = inew
                    nodes[node_kount, 0] = node_kount
                    nodes[node_kount, 1] = xcord[inew]
                    nodes[node_kount, 2] = ycord[inew]
                    nodes[node_kount, 3] = 0.0  # dummy z-coord
                    elem_kount = elem_kount + 1
                    elems[elem_kount, 0] = elem_kount
                    elems[elem_kount, 1] = node_kount_bif
                    elems[elem_kount, 2] = node_kount
                    parent_list[pl_kount] = inew
                    pl_kount = pl_kount + 1

    return node_kount, elem_kount, nodes, elems, continuing, node_map, parent_list


def tellme_figtitle(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


def skel2graph(sk, outputfilename, debug_file, inlet_type, original_image):
    plt.clf()
    plt.imshow(sk)

    plt.setp(plt.gca(), autoscale_on=True)

    tellme_figtitle('Click on, or near to the inlets')

    plt.waitforbuttonpress()
    if inlet_type == 'double':
        pts = plt.ginput(n=2, show_clicks=True, mouse_add=1)
        plt.clf()
    elif inlet_type == 'single':
        pts = plt.ginput(n=1, show_clicks=True, mouse_add=1)
    elif inlet_type == 'TTTS':
        pts = plt.ginput(n=4, show_clicks=True, mouse_add=1)
    else:
        pts = plt.ginput(n=-1, show_clicks=True, mouse_add=1)

    # Im guessing these are coordinates of the umbilical artery insertion
    inlets = np.asarray(pts)
    print('Inlets: ', inlets, len(inlets))

    # converts skeleton to graphical structure
    pixel_graph, coordinates = csr.skeleton_to_csgraph(sk)
    xcord, ycord = coordinates
    len_coord = len(xcord)
    mydegrees = np.zeros(len_coord, dtype=int)
    ne = 0
    closest = np.ones((len(inlets), 2)) * 1000.
    count_isolated = 0
    count_terminal = 0
    count_bifurcations = 0
    count_morefications = 0
    count_continuing = 0
    for i in range(1, len(pixel_graph.indptr) - 1):
        np1 = i - 1  # index for node
        num_attached = (pixel_graph.indptr[i + 1] - pixel_graph.indptr[i])  # looking at how many attachments it has
        if (num_attached == 0):
            count_isolated = count_isolated + 1
            mydegrees[i] = 0
        elif (num_attached == 1):
            count_terminal = count_terminal + 1
            # potental inlet
            for j in range(0, len(inlets)):
                distance = np.sqrt((xcord[i] - inlets[j, 1]) ** 2. + (ycord[i] - inlets[j, 0]) ** 2.)
                if (distance < closest[j, 0]):
                    closest[j, 0] = distance
                    closest[j, 1] = np1
            mydegrees[i] = 1
        elif num_attached == 2:
            count_continuing = count_continuing + 1
            mydegrees[i] = 2
        elif num_attached == 3:
            count_bifurcations = count_bifurcations + 1
            mydegrees[i] = 3
        else:
            count_morefications = count_morefications + 1
            mydegrees[i] = num_attached

    print('closest points to inlet ' + str(closest))
    print('num isolated points ' + str(count_isolated))
    print('num terminals ' + str(count_terminal))
    print('num bifurcations ' + str(count_bifurcations))
    print('num morefiurcations ' + str(count_morefications))

    branch_data = summarize(Skeleton(sk, source_image=original_image))
    node_map = np.zeros(len(xcord))
    nodes = np.zeros((len(xcord), 4))
    elems = np.zeros((len(xcord), 3), dtype=int)
    elem_cnct = np.zeros((len(xcord), 3), dtype=int)
    node_kount = -1
    elem_kount = -1
    for ninlet in range(0, len(inlets)):
        i = int(closest[ninlet, 1]) + 1  # start at the inlet
        node_kount = node_kount + 1
        node_map[node_kount] = i  # old node number
        nodes[node_kount, 0] = node_kount  # new node number
        nodes[node_kount, 1] = xcord[i]  # indexing coordinates array at old node number i
        nodes[node_kount, 2] = ycord[i]
        nodes[node_kount, 3] = 0.0  # dummy z-coordinate
        new_gen_parents = []
        node_kount, elem_kount, nodes, elems, continuing, node_map, branch_parents = new_branch(mydegrees, branch_data,
                                                                                                coordinates,
                                                                                                pixel_graph, i,
                                                                                                node_kount, elem_kount,
                                                                                                nodes, elems, node_map)
        if continuing:
            new_gen_parents = np.append(new_gen_parents, branch_parents[branch_parents > 0])
        cur_gen_parents = new_gen_parents
        while len(cur_gen_parents) > 0:
            new_gen_parents = []
            for k in range(0, len(cur_gen_parents)):
                i = int(cur_gen_parents[k])
                node_kount, elem_kount, nodes, elems, continuing, node_map, branch_parents = new_branch(mydegrees,
                                                                                                        branch_data,
                                                                                                        coordinates,
                                                                                                        pixel_graph, i,
                                                                                                        node_kount,
                                                                                                        elem_kount,
                                                                                                        nodes, elems,
                                                                                                        node_map)
                if continuing:
                    new_gen_parents = np.append(new_gen_parents, branch_parents[branch_parents > 0])
            cur_gen_parents = new_gen_parents

    if debug_file:
        pg.export_exelem_1d(elems[:, :][0:elem_kount + 1], 'arteries', outputfilename)
        pg.export_ex_coords(nodes[:, :][0:node_kount + 1], 'arteries', outputfilename, 'exnode')

    return pixel_graph, coordinates, nodes[:, :][0:node_kount + 1], elems[:, :][0:elem_kount + 1]


def find_branch_points(nodes, elems):
    elem_cnct = pg.element_connectivity_1D(nodes[:, 1:4], elems)
    elem_up = elem_cnct['elem_up']
    elem_down = elem_cnct['elem_down']
    biifurcation_elems = []
    root_node, root_elem = find_root_nodes(nodes, elems)
    bifurcation_nodes = []
    branch_parents = [root_elem[0]]
    terminals = pg.calc_terminal_branch(nodes[:, 1:4], elems)

    for el in range(0, len(elem_down)):
        if elem_down[el][0] == 2:
            biifurcation_elems.append(elems[el, :])
            bif_node = nodes[elems[el][2], :]
            bifurcation_nodes.append(bif_node)
            branch_parents.append(elem_down[el][1])
            branch_parents.append(elem_down[el][2])
    biifurcation_elems = np.asarray(biifurcation_elems)
    branch_end = np.hstack((biifurcation_elems[:, 0], terminals['terminal_elems']))
    return np.asarray(biifurcation_elems), np.asarray(bifurcation_nodes), np.asarray(branch_parents), np.sort(
        branch_end)


def allocate_branch_numbers(nodes, elems):
    bif_elems, bif_nodes, branch_parents, branch_end = find_branch_points(nodes, elems)
    elem_cnct = pg.element_connectivity_1D(nodes[:, 1:4], elems)
    elem_down = elem_cnct['elem_down']
    branch_data = []
    branch = 1
    elem_count = 0
    branch_start = branch_parents[branch - 1]
    branch_structure = np.zeros(len(elems))
    while branch <= len(branch_parents):
        if elem_down[elem_count, 0] == 1:  # element is continuation
            branch_structure[elem_count] = branch
            elem_count = elem_down[elem_count, 1]
        elif elem_down[elem_count, 0] == 2 or elem_down[elem_count, 0] == 0:  # Bifurcation or terminal
            branch_structure[elem_count] = branch
            branch_info = np.asarray([branch, branch_start, elem_count])
            branch_data.append(branch_info)
            if branch == len(branch_parents):  # This will be the last point
                break
            branch += 1
            branch_start = branch_parents[branch - 1]
            elem_count = branch_start

    return branch_structure, np.asarray(branch_data)


def find_middle_index(branch_structure, branch_number):
    # Step 1: Extract indices corresponding to the branch number
    indices = np.where(branch_structure == branch_number)[0]

    # Step 2: Check if indices are found
    if len(indices) == 0:
        return None  # Or raise an exception, or return a default value

    # Step 3: Sort indices
    sorted_indices = np.sort(indices)

    # Step 4: Find the middle index
    middle_index = sorted_indices[len(sorted_indices) // 2]

    return middle_index


def allocate_stem_locations(branch_data, branch_structure, terminals):
    stem_location_elems = []
    terminal_elem = terminals['terminal_elems']

    for i in range(0, len(branch_data)):
        elem_start = branch_data[i, 1]
        elem_end = branch_data[i, 2]
        if elem_start != elem_end:
            if (elem_end not in terminal_elem):
                stem_end_elem = elem_end - 1  # is not terminal so add stem villi before to avoid trifurcation
            else:
                stem_end_elem = elem_end  # is terminal so add stem villi at end
                stem_location_elems.append(stem_end_elem)
            middle_elem = find_middle_index(branch_structure, branch_data[i, 0])
            stem_location_elems.append(middle_elem)

        else:  # Very short branch
            stem_location_elems.append(elem_end)
    return np.asarray(stem_location_elems)


def add_stem_villi(nodes_all, elems_all, sv_length, terminals, radii):
    branch_structure, branch_data = allocate_branch_numbers(nodes_all, elems_all)
    stem_location = allocate_stem_locations(branch_data, branch_structure, terminals)
    print("Number of stem villi added:", len(stem_location))
    node_len = len(nodes_all)
    elem_len = len(elems_all)
    new_node_len = node_len + len(stem_location)
    new_elem_len = elem_len + len(stem_location)
    # initialise new arrays
    chorion_elems = np.zeros((new_elem_len, 3), dtype=int)
    chorion_nodes = np.zeros((new_node_len, 4))
    chorion_radii = np.zeros(new_elem_len)

    node_count = node_len
    elem_count = elem_len
    chorion_nodes[0:node_len, :] = nodes_all
    chorion_elems[0:elem_len, :] = elems_all
    chorion_radii[0:elem_len] = radii
    bif_elems, bif_nodes, branch_parents, branch_end = find_branch_points(nodes_all, elems_all)
    stem_location = stem_location[stem_location != None]
    stem_location = np.unique(stem_location)
    for branch_elem in stem_location:
        connected_node_num = elems_all[branch_elem, 2]
        connected_node = nodes_all[connected_node_num, :]
        if branch_elem not in bif_elems:
            chorion_nodes[node_count][0] = node_count  # Node Number
            chorion_nodes[node_count][1] = connected_node[1]
            chorion_nodes[node_count][2] = connected_node[2]
            chorion_nodes[node_count][3] = connected_node[3] - sv_length
            chorion_elems[elem_count][0] = elem_count
            chorion_elems[elem_count][1] = connected_node_num
            chorion_elems[elem_count][2] = node_count
            chorion_radii[elem_count] = radii[branch_elem]
            elem_count += 1
            node_count += 1
    mask_nodes = np.any(chorion_nodes != 0, axis=1)
    mask_elems = np.any(chorion_elems != 0, axis=1)

    chorion_nodes = chorion_nodes[mask_nodes]
    chorion_elems = chorion_elems[mask_elems]
    chorion_radii = chorion_radii[mask_elems]
    return chorion_nodes, chorion_elems, chorion_radii

def adjust_terminal_branch_radii(nodes, elems, radii, terminals):
    bif_elems, bif_nodes, branch_parents, branch_end = find_branch_points(nodes, elems)
    terminal_elems = terminals['terminal_elems']
    terminal_branches = []
    elem_cnct = pg.element_connectivity_1D(nodes[:, 1:4], elems)
    elem_up = elem_cnct['elem_up']
    for elem_terminal in terminal_elems:
        count = 0
        parent_found = False
        radii_sum = 0
        elem = elem_terminal
        while count <150 and not parent_found:
            radii_sum += radii[elem]
            elem = elem_up[elem][1]
            if elem in branch_parents:
                parent_found = True
            count += 1
        mean_radii = radii_sum/count
        new_count = 0
        elem = elem_terminal
        while new_count != count:
            radii[elem] = mean_radii
            elem = elem_up[elem][1]
            new_count += 1
    return radii
def map_nodes_to_hull(nodes, params, thickness, outputfilename, debug_file):
    z_level = (thickness / 2.0)
    slice_coordinates = params[:, 0]
    for i in range(0, len(nodes)):
        difference = np.abs(slice_coordinates - nodes[i, 2])
        closest_slice_index = np.argmin(difference)
        slice_params = params[closest_slice_index, :]
        scale = 1
        z_closest_ellipse = z_level * np.sqrt(1 - (((nodes[i, 1] - slice_params[4]) / (slice_params[3] * scale)) ** 2))
        while (np.isnan(z_closest_ellipse)):
            scale += 0.01
            z_closest_ellipse = z_level * np.sqrt(
                1 - (((nodes[i, 1] - slice_params[4]) / (slice_params[3] * scale)) ** 2))
        if z_closest_ellipse < nodes[i, 3]:
            nodes[i, 3] = z_closest_ellipse
        else:
            nodes[i, 3] = z_level
    if debug_file:
        pg.export_ex_coords(nodes, 'arteries', outputfilename, 'exnode')
        print('Arterial nodes mapped to shaped hull exported to: ', outputfilename)
    return nodes


def find_root_nodes(nodes, elems):
    node_upstream_end = []
    node_downstream_end = []
    root_nodes = []
    elem_cncty = pg.element_connectivity_1D(nodes[:, 1:4], elems)
    elem_up = elem_cncty['elem_up']
    elem_down = elem_cncty['elem_down']
    for i in range(0, len(elem_down)):
        if (elem_up[i, 0]) == 0:
            node_upstream_end.append(i)
        if (elem_down[i, 0] == 0):
            node_downstream_end.append(i)
    for elements in node_upstream_end:
        node_number = elems[elements, 1]
        root_nodes.append(nodes[node_number, :])

    return np.asarray(root_nodes), np.asarray(node_upstream_end)


def create_umb_anastomosis(nodes, elems, umb_length, output_name, debug_file, inlet_type,radii):
    root_nodes, root_elems = find_root_nodes(nodes, elems)
    if len(root_nodes) == 2 and inlet_type == 'double':
        # calculate coordinates of midpoint for anastomosis
        x_point_1 = root_nodes[0, 1]
        x_point_2 = root_nodes[1, 1]
        y_point_1 = root_nodes[0, 2]
        y_point_2 =  root_nodes[1, 2]

        z_midpoint_1 = root_nodes[0, 3] + (umb_length / 2.0)
        z_point_1 = z_midpoint_1 + umb_length
        z_midpoint_2 = root_nodes[1, 3] + (umb_length / 2.0)
        z_point_2 = z_midpoint_2 + umb_length
        radii_root1 = radii[root_elems[0]]
        radii_root2 = radii[root_elems[1]]
        radii_anast  = (radii_root1 + radii_root2)/2
        node_index = len(nodes)

        # create and append new nodes to end of node file
        new_node_1 = np.asarray([0, x_point_1, y_point_1, z_point_1])
        new_node_1_mid = np.asarray([1, x_point_1, y_point_1, z_midpoint_1])
        new_node_2 = np.asarray([2, x_point_2, y_point_2, z_point_2])
        new_node_2_mid = np.asarray([3, x_point_2, y_point_2, z_midpoint_2])
        new_node_1 = new_node_1.reshape(1, 4)
        new_node_2 = new_node_2.reshape(1, 4)
        new_node_1_mid = new_node_1_mid.reshape(1, 4)
        new_node_2_mid = new_node_2_mid.reshape(1, 4)

        nodes[:, 0] = nodes[:, 0].astype(int) + int(4)
        nodes_combined = np.vstack([new_node_1, new_node_1_mid,new_node_2,new_node_2_mid])
        nodes_new = np.vstack([nodes_combined, nodes])



    # create edit and append elements to match newly created nodes
        elems[:, 0] += 5
        elems[:, 1] += 4
        elems[:, 2] += 4
        in1_anast = np.asarray([0,0,1])
        anast_root1 = np.asarray([1,1,elems[root_elems[0], 1]])
        in2_anast = np.asarray([2,2,3])
        anast_root2 = np.asarray([3,3,elems[root_elems[1], 1]])
        anast = np.asarray([4,1,3])


        in1_anast = in1_anast.reshape(1, 3)
        anast_root1 = anast_root1.reshape(1, 3)
        in2_anast = in2_anast.reshape(1, 3)
        anast_root2 = anast_root2.reshape(1, 3)
        anast = anast.reshape(1, 3)
        elems_new = np.vstack([in1_anast,anast_root1, in2_anast, anast_root2,anast, elems])
        elems_new = elems_new.astype(int)

        new_radii = np.array([radii_root1,radii_root1,radii_root2,radii_root2,radii_anast])
        radii_new = np.concatenate((new_radii,radii))

    elif inlet_type == 'single':
        x_point = root_nodes[0, 1]
        y_point = root_nodes[0, 2]
        z_point = root_nodes[0, 3] + umb_length
        new_node = np.asarray([0, x_point, y_point, z_point])
        new_node = new_node.reshape(1, 4)
        nodes[:, 0] = nodes[:, 0].astype(int) + int(1)
        nodes_new = np.vstack([new_node, nodes])
        # create edit and append elements to match newly created nodes
        elems[:, 0] += 1
        elems[:, 1] += 1
        elems[:, 2] += 1
        anas2anas = np.asarray([0, 0, 1])
        anas2anas = anas2anas.reshape(1, 3)
        elems_new = np.vstack([anas2anas, elems])
        elems_new = elems_new.astype(int)
        radii_new = np.hstack([radii[0],radii])

    pg.export_exelem_1d(elems_new, 'arteries', output_name)
    pg.export_ex_coords(nodes_new, 'arteries', output_name, 'exnode')
    pg.export_exfield_1d_linear(radii_new, 'placenta', 'radii', output_name + '_radii')

    print('Umbilical cord added. Nodes and elems mapped to shaped hull exported to: ', output_name)

    return nodes_new, elems_new, radii_new

def get_vessel_volume(nodes, radii, elems):
    node_1_coords = np.array([nodes[node_id] for node_id in elems[:, 1]])
    node_2_coords = np.array([nodes[node_id] for node_id in elems[:, 2]])
    lengths = np.linalg.norm(node_2_coords[:, 1:4] - node_1_coords[:, 1:4], axis=1)

    vessel_volumes = np.pi * radii ** 2 * lengths
    volume = np.sum(vessel_volumes)

    return volume, vessel_volumes, lengths


def get_euclidean_distance(img):
    euclidean_distance = distance_transform_edt(img)
    return euclidean_distance


def get_radii_from_euclidean(art_nodes, art_elems, euclidean_dist):
    radii_nodes = []
    radii_elems = []
    for nodes in art_nodes:
        radii_nodes.append(euclidean_dist[int(nodes[1]), int(nodes[2])])
    radii_nodes = np.asarray(radii_nodes)
    for elem in art_elems:
        elem_radii = (radii_nodes[elem[1]] + radii_nodes[elem[2]]) / 2
        radii_elems.append(elem_radii)
    return radii_nodes, np.asarray(radii_elems)

def assign_branchID(nodes, elems, branch_data):
    branch_id = np.zeros(len(elems))
    EC = pg.element_connectivity_1D(nodes[:,1:4],elems)
    EC_down = EC['elem_down']
    branch_start = branch_data['branch start']
    branch_end = branch_data['branch end']
    branch_number = 0
    for i in range(0,len(branch_start)):
        ne = branch_start[i]
        branch_id[int(ne)] = i
        while ne != branch_end[i]:
            ne = EC_down[int(ne),1]
            branch_id[int(ne)] = i

    return branch_id
def split_trees(nodes, elems, radii):
    # Create an undirected graph
    G = nx.Graph()
    G.add_edges_from(elems[:, 1:3])  # Add edges using node1 and node2 columns

    # Find connected components
    components = list(nx.connected_components(G))

    if len(components) != 2:
        raise ValueError(f"Expected 2 connected components, but found {len(components)}.")

    # Sort components to consistently assign them as Tree A and Tree B
    components = sorted(components, key=lambda x: min(x))

    # Initialize dictionaries to hold nodes, elements, and radii for each tree
    tree_data = {}

    for i, comp in enumerate(components):
        tree_label = f"tree_{chr(65 + i)}"  # 'tree_A', 'tree_B', etc.

        # Extract nodes belonging to the current component
        comp_nodes_mask = np.isin(nodes[:, 0], list(comp))
        comp_nodes = nodes[comp_nodes_mask]

        # Extract elements where both nodes are in the current component
        comp_elems_mask = np.isin(elems[:, 1], list(comp)) & np.isin(elems[:, 2], list(comp))
        comp_elems = elems[comp_elems_mask]

        # Extract corresponding radii
        comp_radii = radii[comp_elems_mask]

        tree_data[f"{tree_label}_nodes"] = comp_nodes
        tree_data[f"{tree_label}_elems"] = comp_elems
        tree_data[f"{tree_label}_radii"] = comp_radii

    return tree_data
def reindex_tree(nodes, elems, radii):
    """
    Reindex the nodes and elements of a tree to start from 0 and update elements accordingly.

    Parameters:
    - nodes: np.ndarray of shape (n, 4) with columns [node_id, x, y, z]
    - elems: np.ndarray of shape (m, 3) with columns [elem_id, node1, node2]
    - radii: np.ndarray of shape (m, 1) corresponding to each element's radius

    Returns:
    - new_nodes: np.ndarray with updated node indices
    - new_elems: np.ndarray with updated element indices and node references
    - new_radii: np.ndarray corresponding to the elements
    """

    # Extract original node and element IDs
    original_node_ids = nodes[:, 0]
    original_elem_ids = elems[:, 0]

    # Create a mapping from original node IDs to new indices starting from 0
    node_id_to_new_index = {old_id: new_index for new_index, old_id in enumerate(original_node_ids)}

    # Apply the mapping to the nodes array
    new_nodes = nodes.copy()
    new_nodes[:, 0] = np.array([node_id_to_new_index[old_id] for old_id in original_node_ids])

    # Apply the mapping to the elements array for node references
    new_elems = elems.copy()
    new_elems[:, 1] = np.array([node_id_to_new_index[old_id] for old_id in elems[:, 1]])
    new_elems[:, 2] = np.array([node_id_to_new_index[old_id] for old_id in elems[:, 2]])

    # Reindex element IDs to start from 0
    new_elems[:, 0] = np.arange(len(new_elems))

    # Radii remain unchanged but ensure it's a copy to maintain consistency
    new_radii = radii.copy()

    return new_nodes, new_elems, new_radii


def chorion_branching_analytics(trees,sample_number,export_directory, inlet_type, export_data):
    arterial_geom_B = dict.fromkeys(['nodes', 'elems', 'radii', 'length', 'branch id'])
    arterial_geom_A = dict.fromkeys(['nodes', 'elems', 'radii', 'length', 'branch id'])

    if inlet_type == 'double':
        # TREE A
        arterial_geom_A['nodes'] = trees['tree_A_nodes']
        arterial_geom_A['elems'] = trees['tree_A_elems']
        arterial_geom_A['radii'] = trees['tree_A_radii']
        arterial_geom_A['length'] = pg.define_elem_lengths(trees['tree_A_nodes'][:, 1:4], trees['tree_A_elems'])

        art_branch_data = pg.define_branch_from_geom(arterial_geom_A)
        branch_id = assign_branchID(trees['tree_A_nodes'], trees['tree_A_elems'], art_branch_data)
        arterial_geom_A['branch id'] = branch_id

        branch_geom = {}
        branch_geom['nodes'] = trees['tree_A_nodes']
        branch_geom['elems'] = art_branch_data['elems']
        branch_geom['euclidean length'] = pg.define_elem_lengths(trees['tree_A_nodes'][:, 1:4], art_branch_data['elems'])
        arterial_geom_A, branch_geom_A, generation_table_A, strahler_table_A, branch_table_A = pg.analyse_branching(
            arterial_geom_A,
            branch_geom,
            'strahler',
            1., 1.)

        tree_B_nodes_reindexed, tree_B_elems_reindexed, tree_B_radii_reindexed = reindex_tree(trees['tree_B_nodes'], trees['tree_B_elems'],
                                                                                              trees['tree_B_radii'])

        # TREE B
        arterial_geom_B['nodes'] = tree_B_nodes_reindexed
        arterial_geom_B['elems'] = tree_B_elems_reindexed
        arterial_geom_B['radii'] = tree_B_radii_reindexed
        arterial_geom_B['length'] = pg.define_elem_lengths(tree_B_nodes_reindexed[:, 1:4], tree_B_elems_reindexed)

        art_branch_data_B = pg.define_branch_from_geom(arterial_geom_B)
        branch_id_B = assign_branchID(tree_B_nodes_reindexed, tree_B_elems_reindexed, art_branch_data_B)
        arterial_geom_B['branch id'] = branch_id_B

        branch_geom_B = {}
        branch_geom_B['nodes'] = tree_B_nodes_reindexed
        branch_geom_B['elems'] = art_branch_data_B['elems']
        branch_geom_B['euclidean length'] = pg.define_elem_lengths(tree_B_nodes_reindexed[:, 1:4], art_branch_data_B['elems'])
        arterial_geom_B, branch_geom_B, generation_table_B, strahler_table_B, branch_table_B = pg.analyse_branching(
            arterial_geom_B,
            branch_geom_B,
            'strahler',
            1., 1.)

        # csv files
        if export_data:
            print('Writing files')
            output = export_directory+sample_number+'_A_'+ 'StrahlerTable.csv'
            headerTable = "'Order', 'NumBranches', 'Length(mm)', 'std', 'Diameter(mm)', 'std', 'EuclideanLength(mm)', 'std', 'Len/Diam', 'std', 'Tortuosity', 'std', 'Angles', 'std', 'LenRatio', 'std', 'DiamRatio', 'std'"
            np.savetxt(output, strahler_table_A, fmt='%.4f', delimiter=',', header=headerTable)

            output = export_directory+sample_number+'_A_'+ 'GenerationTable.csv'
            headerTable = "'Gen', 'NumBranches', 'Length(mm)', 'std', 'Diameter(mm)', 'std', 'Euclidean Length(mm)', 'std', 'Len/Diam', 'std', 'Tortuosity', 'std', 'Angles', 'std', 'Minor Angle', 'std', 'Major Angle', 'std', 'LLparent', 'std', 'LminLparent', 'std', 'LmajLparent', 'std', 'LminLmaj', 'std', 'DDparent', 'std', 'DminDparent', 'std', 'DmajDparent', 'std', 'DminDmaj', 'std'"
            np.savetxt(output, generation_table_A, fmt='%.4f', delimiter=',', header=headerTable)

            output = export_directory +sample_number+'_A_'+ 'OverallTable.csv'
            headerTable = "'Num branches', 'Total length','Total vessel volume', 'Total volume', 'vascular span','inlet diameter','num generations', 'num orders', 'ave term gen','std','tortuosity','std','branch length', 'std', 'euc length', 'std', 'diameter','std','L/D','std', 'branch angle', 'std','minor angle','std', 'major angle', 'std', 'D/Dparent', 'std', 'Dmin/Dparent','std', 'Dmaj/Dparent', 'std', 'L/Lparent', 'std','L/Lparent','std', 'Lmin/Lparent','std','Lmaj/Lparent', 'std', 'Lmaj/Lmin','std', 'Rb', 'rsq','Rd','rsq','Rl','rsq'"
            np.savetxt(output, branch_table_A, fmt='%.4f', delimiter=',', header=headerTable)

            output = export_directory +sample_number+'_B_'+ 'StrahlerTable.csv'
            headerTable = "'Order', 'NumBranches', 'Length(mm)', 'std', 'Diameter(mm)', 'std', 'EuclideanLength(mm)', 'std', 'Len/Diam', 'std', 'Tortuosity', 'std', 'Angles', 'std', 'LenRatio', 'std', 'DiamRatio', 'std'"
            np.savetxt(output, strahler_table_B, fmt='%.4f', delimiter=',', header=headerTable)

            output = export_directory +sample_number+'_B_'+ 'GenerationTable.csv'
            headerTable = "'Gen', 'NumBranches', 'Length(mm)', 'std', 'Diameter(mm)', 'std', 'Euclidean Length(mm)', 'std', 'Len/Diam', 'std', 'Tortuosity', 'std', 'Angles', 'std', 'Minor Angle', 'std', 'Major Angle', 'std', 'LLparent', 'std', 'LminLparent', 'std', 'LmajLparent', 'std', 'LminLmaj', 'std', 'DDparent', 'std', 'DminDparent', 'std', 'DmajDparent', 'std', 'DminDmaj', 'std'"
            np.savetxt(output, generation_table_B, fmt='%.4f', delimiter=',', header=headerTable)

            output = export_directory +sample_number+'_B_'+ 'OverallTable.csv'
            headerTable = "'Num branches', 'Total length','Total vessel volume', 'Total volume', 'vascular span','inlet diameter','num generations', 'num orders', 'ave term gen','std','tortuosity','std','branch length', 'std', 'euc length', 'std', 'diameter','std','L/D','std', 'branch angle', 'std','minor angle','std', 'major angle', 'std', 'D/Dparent', 'std', 'Dmin/Dparent','std', 'Dmaj/Dparent', 'std', 'L/Lparent', 'std','L/Lparent','std', 'Lmin/Lparent','std','Lmaj/Lparent', 'std', 'Lmaj/Lmin','std', 'Rb', 'rsq','Rd','rsq','Rl','rsq'"
            np.savetxt(output, branch_table_B, fmt='%.4f', delimiter=',', header=headerTable)
    elif inlet_type == 'single':
        arterial_geom_A = dict.fromkeys(['nodes', 'elems', 'radii', 'length', 'branch id'])
        arterial_geom_A['nodes'] = trees['tree_A_nodes']
        arterial_geom_A['elems'] = trees['tree_A_elems']
        arterial_geom_A['radii'] = trees['tree_A_radii']
        arterial_geom_A['length'] = pg.define_elem_lengths(trees['tree_A_nodes'][:, 1:4], trees['tree_A_elems'])

        art_branch_data = pg.define_branch_from_geom(arterial_geom_A)
        branch_id = assign_branchID(trees['tree_A_nodes'], trees['tree_A_elems'], art_branch_data)
        arterial_geom_A['branch id'] = branch_id
    return arterial_geom_A, arterial_geom_B

def assign_branchID(nodes, elems, branch_data):
    branch_id = np.zeros(len(elems))
    EC = pg.element_connectivity_1D(nodes[:,1:4],elems)
    EC_down = EC['elem_down']
    branch_start = branch_data['branch start']
    branch_end = branch_data['branch end']
    branch_number = 0
    for i in range(0,len(branch_start)):
        ne = branch_start[i]
        branch_id[int(ne)] = i
        while ne != branch_end[i]:
            ne = EC_down[int(ne),1]
            branch_id[int(ne)] = i

    return branch_id

def define_geom(nodes,elems,radii):
    trees = {}
    trees['tree_A_nodes'] = nodes
    trees['tree_A_elems'] = elems
    trees['tree_A_radii'] = radii
    return trees

def get_inlet_branch_radius(geom):
    nodes = geom['nodes']
    elems = geom['elems']
    radii = geom['radii']
    branch_id = geom['branch id']
    inlet_node, inlet_elem = find_root_nodes(nodes,elems)
    branch_number_inlet = branch_id[inlet_elem[0]]
    inlet_branch_mask = branch_id==branch_number_inlet
    inlet_branch_elems = elems[inlet_branch_mask]
    inlet_branch_radii = radii[inlet_branch_mask]
    radius  = np.max(inlet_branch_radii)
    return radius

def set_inlet_branch_radius(geom, radius):
    nodes = geom['nodes']
    elems = geom['elems']
    radii = geom['radii']
    branch_id = geom['branch id']
    inlet_node, inlet_elem = find_root_nodes(nodes,elems)
    branch_number_inlet = branch_id[inlet_elem[0]]
    for i in range(0,len(elems)):
        if branch_number_inlet == branch_id[i]:
            radii[i] = radius
    geom['radii'] = radii
    return geom

def recombine_trees(Geom_A,Geom_B):
    """
    Recombine two trees by offsetting IDs in tree_B to avoid conflicts.

    Parameters:


    Returns:
    - combined_nodes: np.ndarray of shape (n_A + n_B, 4)
    - combined_elems: np.ndarray of shape (m_A + m_B, 3)
    - combined_radii: np.ndarray of shape (m_A + m_B, 1)
    """
    tree_A_nodes = Geom_A['nodes']
    tree_A_elems = Geom_A['elems']
    tree_A_radii = Geom_A['radii']
    tree_B_nodes = Geom_B['nodes']
    tree_B_elems = Geom_B['elems']
    tree_B_radii = Geom_B['radii']
    # Determine the maximum IDs in tree_A
    max_node_id_A = int(np.max(tree_A_nodes[:, 0]))
    max_elem_id_A = int(np.max(tree_A_elems[:, 0]))

    # Offset for tree_B IDs
    node_id_offset = max_node_id_A + 1
    elem_id_offset = max_elem_id_A + 1

    # Update node IDs in tree_B
    tree_B_nodes_updated = tree_B_nodes.copy()
    tree_B_nodes_updated[:, 0] += node_id_offset

    # Create a mapping from old to new node IDs
    old_to_new_node_ids = {
        old_id: new_id for old_id, new_id in zip(
            tree_B_nodes[:, 0], tree_B_nodes_updated[:, 0]
        )
    }

    # Update element IDs and node references in tree_B
    tree_B_elems_updated = tree_B_elems.copy()
    tree_B_elems_updated[:, 0] += elem_id_offset
    tree_B_elems_updated[:, 1] = np.array([
        old_to_new_node_ids.get(node_id, node_id) for node_id in tree_B_elems[:, 1]
    ])
    tree_B_elems_updated[:, 2] = np.array([
        old_to_new_node_ids.get(node_id, node_id) for node_id in tree_B_elems[:, 2]
    ])

    # Concatenate nodes, elements, and radii
    combined_nodes = np.concatenate((tree_A_nodes, tree_B_nodes_updated), axis=0)
    combined_elems = np.concatenate((tree_A_elems, tree_B_elems_updated), axis=0)
    combined_radii = np.concatenate((tree_A_radii, tree_B_radii), axis=0)

    return combined_nodes, combined_elems, combined_radii


def find_parent_list(nodes, elems):
    node_downstream_end = []
    parent_nodes = []
    elem_cncty = element_connectivity_multi(nodes[:, 1:4], elems)
    elem_up = elem_cncty['elem_up']
    elem_down = elem_cncty['elem_down']
    for i in range(0, len(elem_down)):
        if (elem_down[i, 0] == 0):
            node_downstream_end.append(i)
    for elements in node_downstream_end:
        node_number = elems[elements, 1]
        parent_nodes.append(nodes[node_number, :])
    return np.asarray(parent_nodes), np.asarray(node_downstream_end)

def element_connectivity_multi(node_loc, elems):
    # Initialise connectivity arrays
    anastomosis = False
    num_elems = len(elems)
    num_nodes = len(node_loc)
    elems_at_node = np.zeros((num_nodes, 20), dtype=int) #allow up to 20-furcations
    # determine elements that are associated with each node
    for ne in range(0, num_elems):
        for nn in range(1, 3):
            nnod = elems[ne][nn]
            elems_at_node[nnod][0] = elems_at_node[nnod][0] + 1
            elems_at_node[nnod][elems_at_node[nnod][0]] = ne
    elem_upstream = np.zeros((num_elems, int(np.max(elems_at_node[:,0]))), dtype=int)
    elem_downstream = np.zeros((num_elems, int(np.max(elems_at_node[:,0]))), dtype=int)
    # assign connectivity
    for ne in range (0,num_elems):
        upstream_node = elems[ne][1]
        downstream_node = elems[ne][2]
        nnod2 = elems[ne][2]  # second node in elem
        if elems_at_node[nnod2][0] == 3:
            elem1 = elems_at_node[nnod2][1]
            elem2 = elems_at_node[nnod2][2]
            elem3 = elems_at_node[nnod2][3]
            if (elems[elem1, 2] == elems[elem2, 2]) or (elems[elem2, 2] == elems[elem3, 2]) or (
                    elems[elem1, 2] == elems[elem3, 2]):
                anastomosis = True
        for noelem in range(1, elems_at_node[nnod2][0] + 1):
            ne2 = elems_at_node[nnod2][noelem]

            if ne2 != ne:
                if not anastomosis:
                    elem_upstream[ne2][0] = elem_upstream[ne2][0] + 1
                    elem_upstream[ne2][elem_upstream[ne2][0]] = ne
                    elem_downstream[ne][0] = elem_downstream[ne][0] + 1
                    elem_downstream[ne][elem_downstream[ne][0]] = ne2
                elif anastomosis:
                    anastomosis = False
                    if elems[ne][2] != elems[ne2][2]:
                        elem_upstream[ne2][0] = elem_upstream[ne2][0] + 1
                        elem_upstream[ne2][elem_upstream[ne2][0]] = ne
                        elem_downstream[ne][0] = elem_downstream[ne][0] + 1
                        elem_downstream[ne][elem_downstream[ne][0]] = ne2
    return {'elem_up': elem_upstream, 'elem_down': elem_downstream}

def set_radii_per_parent(tree, parent_list_nodes, parent_list_elems, radii,terminal_villi_radius):
    nodes = tree['nodes']
    elems = tree['elems']
    orders = pg.evaluate_orders(nodes[:,1:4],elems)
    strahler_order = orders['strahler']
    EC = pg.element_connectivity_1D(nodes[:,1:4],elems)
    elem_down = EC['elem_down']
    print(f"Adjusting radii from {len(parent_list_elems)} parents")
    # --- Usage ---
    radii_downstream = np.zeros(len(elems))
    radii_downstream[0:len(radii)] = radii
    elem_downstream = build_downstream_dict(elems, elem_down)

    for parent_elem in parent_list_elems:
        parent_strahler = strahler_order[parent_elem]
        parent_radius = radii[parent_elem]
        strahler_ratio = (parent_radius/terminal_villi_radius)**(1/(parent_strahler-1))

        downstream = get_all_downstream_elements(parent_elem, elem_downstream)
        print(f"Adjusting radius for parent elem {parent_elem} with {len(downstream)} children elements")
        print(f"Parent strahler is {parent_strahler}, Stahler ratio for child vessels is {strahler_ratio}")
        for elem_num in downstream:
            radii_downstream[elem_num] = parent_radius/(strahler_ratio **(parent_strahler-strahler_order[elem_num]))

    return radii_downstream

def build_downstream_dict(elements, downstream_array):
    """
    Parameters:
    -----------
    elements : np.ndarray, shape (N, 3)
        Columns: [element_id, node1, node2]
    downstream_array : np.ndarray, shape (N, 3)
        Columns: [num_downstream, downstream_elem_1, downstream_elem_2]
        Index corresponds to element array rows.
        Terminal elements: [0, 0, 0]
        Single downstream: [1, next_elem, 0]
        Bifurcation: [2, next_elem_1, next_elem_2]
    """
    elem_downstream = {}

    for i, row in enumerate(elements):
        elem_id = int(row[0])
        num_down = int(downstream_array[i, 0])

        if num_down == 0:
            elem_downstream[elem_id] = []          # terminal element
        elif num_down == 1:
            elem_downstream[elem_id] = [int(downstream_array[i, 1])]
        elif num_down == 2:
            elem_downstream[elem_id] = [int(downstream_array[i, 1]),
                                        int(downstream_array[i, 2])]

    return elem_downstream


def get_all_downstream_elements(start_element, elem_downstream):
    """
    BFS traversal to collect all downstream elements from start_element.
    Returns list excluding the start element itself.
    """
    from collections import deque

    visited = set()
    queue = deque(elem_downstream.get(start_element, []))

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for next_elem in elem_downstream.get(current, []):
            if next_elem not in visited:
                queue.append(next_elem)

    return list(visited)

def reassign_radii(geom, parent_list_nodes, parent_list_elems, radii, terminal_radius):
    nodes = geom['nodes']
    elems = geom['elems']
    elem_down = geom['elem_down']

    return