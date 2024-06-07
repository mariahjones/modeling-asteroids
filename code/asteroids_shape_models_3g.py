import numpy as np
import copy
import pandas as pd
import astropy as ap
from scipy.stats import circmean
from astropy.constants import L_sun
from spt3g import core, util
from spt3g.pointing.offline_pointing import GreatCircleDistance
from spt3g.core import G3Units
from spt3g.util import thermo

from scipy.constants import sigma, h, k, c


# ==================================================================================
# helper functions
# ==================================================================================

def get_color(val, vmin, vmax, cmap="viridis"):
    """
    Get matplotlib color within some chosen scale.

    Arguments:
    ----------
    val : float
        Value to plot
    vmin : float
        Minimum value of color range
    vmax : float
        Maximum value of color range
    cmap : str ["viridis"]
        Matplotlib colormap name to use
    
    Returns:
    --------
        hex color to plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mpc
    from matplotlib import cm

    # get hex color for plotting multiple times
    cmap = plt.get_cmap(cmap)
    color_norm = mpc.Normalize(vmin=vmin, vmax=vmax)
    scalarmap = cm.ScalarMappable(norm=color_norm, cmap=cmap)
    if val > vmax:
        val = vmax
    if val < vmin:
        val = vmin
    rgb = scalarmap.to_rgba(val)
    return mpc.rgb2hex(rgb)

def Rz(theta):
    """
    Make a matrix that rotates about the z axis

    Arguments:
    ----------
    theta : float
        angle in radians to rotate by
    
    Returns:
    --------
    np.ndarray
        3x3 matrix such that v_rotated = matrix * v_unrotated
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.],
        [0., 0., 1.]
    ])

def Ry(theta):
    """
    Make a matrix that rotates about the y axis

    Arguments:
    ----------
    theta : float
        angle in radians to rotate by
    
    Returns:
    --------
    np.ndarray
        3x3 matrix such that v_rotated = matrix * v_unrotated
    """
    return np.array([
        [np.cos(theta), 0., np.sin(theta)],
        [0., 1., 0.],
        [-np.sin(theta), 0., np.cos(theta)]
    ])

def xyz2rae(xyz):
    """
    Converts (x, y, z) to (r, az, el) coords

    Arguments:
    ----------
    xyz : listlike or listlike of listlike
        (x, y, z) or list of (x, y, z) coordinates to transform
    
    Returns:
    --------
    np.array
        spherical coords in same dimension as the argument
    """

    # format args
    pts = np.array(np.hstack((xyz, np.zeros_like(xyz))), dtype=float)
    d = pts.ndim
    if d == 1:
        pts = np.array([pts])

    # do the calculations
    xy = pts[:,0]**2 + pts[:,1]**2
    pts[:,3] = np.sqrt(xy + pts[:,2]**2) # r = X^2 + Y^2 + Z^2
    pts[:,4] = np.arctan2(pts[:,1], pts[:,0]) # az = atan(Y / X)
    pts[:,5] = np.arctan2(pts[:,2], np.sqrt(xy)) # el = atan(Z / XYplaneR)

    # return in appropriate dimensions
    if d == 1:
        return pts[0,3:]
    return pts[:,3:]

def rae2xyz(rae):
    """
    Converts (r, az, el) to (x, y, z) coords

    Arguments:
    ----------
    rae : listlike or listlike of listlike
        (r, az, el) or list of (r, az, el) coordinates to transform
    
    Returns:
    --------
    np.array
        cartesian coords in same dimension as the argument
    """

    # format args
    pts = np.array(np.hstack((rae, np.zeros_like(rae))), dtype=float)
    d = pts.ndim
    if d == 1:
        pts = np.array([pts])

    # do the calculations
    pts[:,3] = pts[:,0] * np.cos(pts[:,2]) * np.cos(pts[:,1]) # x = r * cosel * cosaz
    pts[:,4] = pts[:,0] * np.cos(pts[:,2]) * np.sin(pts[:,1]) # y = r * cosel * sinaz
    pts[:,5] = pts[:,0] * np.sin(pts[:,2]) # z = r * sinel
    
    # return in appropriate dimensions
    if d == 1:
        return pts[0,3:]
    return pts[:,3:]

def ray_triangle_intersection(ray_start, ray_vec, triangle):
    """
    Moeller–Trumbore intersection algorithm.

    Parameters
    ----------
    ray_start : np.ndarray
        Length three numpy array representing start of point.
    ray_vec : np.ndarray
        Direction of the ray.
    triangle : np.ndarray
        3x3 numpy array containing the three vertices of a triangle.

    Returns
    -------
    bool
        True when there is an intersection.
    tuple
        Length three tuple containing the distance ``t``, and the intersection in unit
        triangle ``U``, ``V`` coordinates.  When there is no intersection, these values
        will be: ``[np.nan, np.nan, np.nan]``
    """
    # define a null intersection
    null_inter = np.array([np.nan, np.nan, np.nan])

    # break down triangle into the individual points
    v1, v2, v3 = triangle
    eps = 0.000001

    # compute edges
    edge1 = v2 - v1
    edge2 = v3 - v1
    pvec = np.cross(ray_vec, edge2)
    det = edge1.dot(pvec)

    if abs(det) < eps:  # no intersection
        return False, null_inter
    inv_det = 1.0 / det
    tvec = ray_start - v1
    U = tvec.dot(pvec) * inv_det

    if U < 0.0 or U > 1.0:  # if not intersection
        return False, null_inter

    qvec = np.cross(tvec, edge1)
    V = ray_vec.dot(qvec) * inv_det
    if V < 0.0 or U + V > 1.0:  # if not intersection
        return False, null_inter

    t = edge2.dot(qvec) * inv_det
    if t < eps:
        return False, null_inter

    return True, np.array([t, U, v])

def check_2Dtri_winding(tri, allow_reversed=True):
    """
    Check the orientation of a triangle on XY plane and mixes order to match RH rule
    From: https://rosettacode.org/wiki/Determine_if_two_triangles_overlap#Python

    Parameters
    ----------
    tri: list-like
        List of 3 [x,y] coordinates of the triangle
    allow_reversed: bool [True]
        Whether to allow a LH-oriented triangle. If False and a LH-oriented tri is
        provided, raises an error. If True, returns the same triangle in RH order. 

    Returns
    -------
    trisq: np.array
        Array of [x,y] points in RH orientation
    """
    trisq = np.ones((3,3))
    trisq[:,0:2] = np.array(tri)
    detTri = np.linalg.det(trisq)
    if detTri < 0.0:
        if allow_reversed:
            a = trisq[2,:].copy()
            trisq[2,:] = trisq[1,:]
            trisq[1,:] = a
        else: raise ValueError("triangle has wrong winding direction")
    return trisq

def check_2Dtris_overlap(t1, t2, eps=0.0, allow_reversed=True, on_boundary=False):
    """
    Check the orientation of a triangle on XY plane and mixes order to match RH rule
    From: https://rosettacode.org/wiki/Determine_if_two_triangles_overlap#Python

    Arguments:
    ----------
    t1, t2: list-like
        Lists of 3 [x,y] coordinates of the triangles
    eps: float [0.0]
        Tolerance for how close counts as touching
    allow_reversed: bool [True]
        Whether to allow a LH-oriented triangle. If False and a LH-oriented tri is
        provided, raises an error.
    on_boundary: bool [False]
        Whether points on the boundary of the triangle count as overlapping

    Returns:
    --------
    bool
        Whether or not the triangles are overlapping
    """
    #Trangles must be expressed anti-clockwise
    t1s = check_2Dtri_winding(t1, allow_reversed)
    t2s = check_2Dtri_winding(t2, allow_reversed)

    if on_boundary:
        #Points on the boundary are considered as colliding
        check_edge = lambda x: np.linalg.det(x) < eps
    else:
        #Points on the boundary are not considered as colliding
        check_edge = lambda x: np.linalg.det(x) <= eps

    #For edge E of trangle 1,
    for i in range(3):
        edge = np.roll(t1s, i, axis=0)[:2,:]

        #Check all points of trangle 2 lay on the external side of the edge E. If
        #they do, the triangles do not collide.
        if (check_edge(np.vstack((edge, t2s[0]))) and
            check_edge(np.vstack((edge, t2s[1]))) and  
            check_edge(np.vstack((edge, t2s[2])))):
            return False

    #For edge E of trangle 2,
    for i in range(3):
        edge = np.roll(t2s, i, axis=0)[:2,:]

        #Check all points of trangle 1 lay on the external side of the edge E. If
        #they do, the triangles do not collide.
        if (check_edge(np.vstack((edge, t1s[0]))) and
            check_edge(np.vstack((edge, t1s[1]))) and  
            check_edge(np.vstack((edge, t1s[2])))):
            return False

    #The triangles collide
    return True


# ==================================================================================
# shape modeling class
# ==================================================================================

class AsteroidShapeModel:
    """
    Shape model for an asteroid.
    """
    
    def __init__(self, asteroid=None, shape_file=None, spin_file=None,  convex=True,
                 calibrated_size=True):
        """
        Initialize an asteroid shape model.
        
        Arguments:
        ----------
        asteroid : str or int [None]
            If provided, the name of the asteroid to query DAMIT. Will download spin.txt
            and shape.obj data files.  Must provide either shape_file or asteroid. Can
            be either the asteroid's name (ex: "Ceres"), number (ex: 1), temporary
            designation (ex: "2001 BB26"), or "DAMIT_{modelnumber}" to directly use a
            known model (ex: "damit_5915"). Case insensitive.
        shape_file : str [None]
            If provided, the name of the shape file in DAMIT format to read. Must
            provide the generated shape.obj file, NOT the shape.txt file.  Must  provide
            either shape_file or asteroid.
        spin_file : str [None]
            If provided, the name of the spin file in DAMIT format to read. Must provide
            the generated spin.txt file, NOT the IAUspin.txt file.
        convex : bool [True]
            Whether the shape model is a convex polyhedron. Used only if shape file is
            provided, otherwise read from DAMIT while loading files.
        calibrated_size : bool [True]
            Whether the shape model is scaled to its real size in km (True) or is scaled
            to unit volume (False). Used only if shape file is provided, otherwise read
            from DAMIT while loading files.  If False, consider running calibrate_size()
            with equivalent spherical diameter.
        """
        # initialize all variables:
        # reference info
        self.asteroid = asteroid # name of obj. if pulled from DAMIT, in priority: des, name, num
        self.asteroid_id = None # DAMIT asteroid ID
        self.model_id = None # DAMIT model ID
        self.spin_url = None # url to DAMIT generated spin.txt data file
        self.shape_url = None # url to DAMIT generated shape.obj data file

        # shape-relevant variables
        self.convex = convex # bool whether shape is convex
        self.calibrated_size = calibrated_size # bool whether size is calibrated (i.e. vertex coords in length) or not (i.e. unitary volume)
        self.nvertices = None # number of vertices in model
        self.vertices_init = None # list of (X, Y, Z) coords of vertices in co-rotating coord frame (in g3units if calibrated_size)
        self.vertices = None # list of (X, Y, Z) coords of vertices in ecliptic coord frame (in g3units if calibrated_size)
        self.nfacets = None # number of facets in model
        self.facets = None # list of 3 vertex indeces that compose facets, in CCW order from outside
        self.norms_init = None # list of (X, Y, Z) coords of facet unit normal vectors in ecliptic coord frame
        self.norms = None # list of (X, Y, Z) coords of facet unit normal vectors in ecliptic coord frame
        self.areas = None # list of areas of each facet (in G3Units if calibrated_size)
        self.volume = None # total volume of model (in G3Units if calibrated_size)
        self.diameter = None # volume-equivalent spherical diameter

        # spin-relevant variables
        self.spin_long = None # spin axis ecliptic longitude
        self.spin_lat = None # spin axis ecliptic latitude
        self.spin_rate = None # angle units rotated per time unit
        self.period = None # sidereal rotation period
        self.spin_epoch = None # JD epoch for measured spin axis orientation
        self.spin_offset = None # spin rotation at self.spin_epoch
        self.spin_axis_init = None # unit vector direction of rotational axis in co-rotating coord frame
        self.spin_axis = None # unit vector direction of rotational axis in ecliptic coord frame
        self.x_axis = np.array([1., 0., 0.]) # unit vector direction of body-centric x axis in ecliptic coord frame
        self.y_axis = np.array([0., 1., 0.]) # unit vector direction of body-centric y axis in ecliptic coord frame

        # position-relevant variables
        self.epoch = None # JD epoch of current vertices and norms
        self.loc = None # vector (X, Y, Z) coord of asteroid in ecliptic coords
        #self.earth_loc = None # vector (X, Y, Z) coord of Earth in ecliptic coords
        self.sun_loc = None # vector (X, Y, Z) coord of Sun in equatorial coord from obs site
        self.earth_loc = None # vector (X, Y, Z) coord of asteroid in equatorial coord from obs site
        self.sun = None # unit vector direction of the sun
        self.sun_corot = None # unit vector direction of the sun in co-rotating frame
        self.sun_dist = None # distance to the sun
        self.solar_intensity = None # incident solar intensity (power/area) at current solar distance
        self.cos_sun = None # cos of angle between each facet norm and sun vector
        self.sunlit = None # whether each facet is (True) or is not (False) lit
        self.observer = None # unit vector direction of the observer
        self.observer_corot = None # unit vector direciton of the observer in co-rotating frame
        self.observer_dist = None # distance to the observer
        self.cos_obs = None # cos of angle between each facet norm and observer vector
        self.visible = None # whether each facet is (1) or is not (0) observable
        self.precalculated_time_step = None # time length of each step in one full rotation
        self.precalculated_cos_sun = None # calculates sunlit facets at each nth step
        self.precalculated_n_steps = None # number of steps to take in one full rotation

        # thermal-relevant properties and variables
        self.albedo = None # albedo
        self.eps = None # emissivity
        #self.Rv = None # Fesnel reflection loss at the surface interface
        self.cond = None # thermal conductivity k in G3Units
        self.therm_inert = None # thermal inertia sqrt(k*rho*c) in G3Units
        self.Ls = None # thermal skin depth (reciprocal of absorption coefficient)
        self.depths = None # array of depths below subsurface sampled in simulation, ordered in increasing depth
        self.depth = None # maximum depth below subsurface sampled in simulation
        self.thicknesses = None # array of how thick each depth layer is
        self.thickness = None # for equal thickness depth layers, the thickness of each depth
        self.temps = None # array of temperatures for each facet at each skin depth, shape (len(depths), nfacets)
        
        # search DAMIT for links to data files if requested
        if (asteroid is not None) and (shape_file is None):

            # read in info on available DAMIT models
            ast = None
            model = None
            taburl = "https://astro.troja.mff.cuni.cz/projects/damit/exports/table/"
            try:
                models = pd.read_csv(taburl + "asteroid_models")
                asts = pd.read_csv(taburl + "asteroids")
            except:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                models = pd.read_csv(taburl + "asteroid_models")
                asts = pd.read_csv(taburl + "asteroids")

            # get asteroid ID info if specific DAMIT model number provided
            self.asteroid = str(asteroid).upper()
            if self.asteroid.startswith('DAMIT_'):
                self.model_id = int(self.asteroid.split('_')[-1])
                model = models[models["id"] == self.model_id].iloc[0]
                self.asteroid_id = int(model["asteroid_id"])
                ast = asts[asts['id'] == self.asteroid_id]

            # otherwise search available models for requested asteroid
            else:
                try:
                    names_upper = np.array([str(name).upper() for name in asts['name']])
                    des_upper = np.array(
                        [str(name).upper() for name in asts['designation']]
                    )
                    if self.asteroid in names_upper: # arg is name
                        ast = asts[names_upper == self.asteroid]
                    elif self.asteroid in des_upper: # arg is designation
                        ast = asts[des_upper == self.asteroid]
                    elif np.any(float(self.asteroid) == asts['number']): # arg is num
                        ast = asts[float(self.asteroid) == asts['number']]
                    else:
                        raise Exception()
                    self.asteroid_id = int(ast['id'].item())
                except:
                    raise ValueError("Invalid asteroid name")
                
                # select most recent DAMIT model
                model = models[models['asteroid_id'] == self.asteroid_id]
                model = model[model['version'] == np.max(model['version'])]
                if len(model) == 0: # no model match, model is likely a tumbler model
                    raise NotImplementedError("No non-tumbler model found for asteroid id %s."%self.asteroid_id)
                if len(model) > 1:
                    import warnings
                    model = model.iloc[-1]
                    warnings.warn(
                        "Multiple DAMIT models found with same version date. Using model id %s. Consider supplying desired model files or DAMIT model number."%model['id']
                    )
                else:
                    model = model.iloc[0]
                self.model_id = int(model["id"])
            
            # update asteroid query name
            if str(ast["designation"].item()) != 'nan':
                self.asteroid = str(ast["designation"].item())
            elif str(ast["name"].item()) != 'nan':
                self.asteroid = str(ast["name"].item())
            elif not np.isnan(ast["number"].item()):
                self.asteroid = str(int(ast["number"].item()))
            
            # save rest of info and clean up
            self.convex = model["nonconvex"] != 1
            self.calibrated_size = model["calibrated_size"] == 1
            modurl = "https://astro.troja.mff.cuni.cz/projects/damit/generated_files/open/AsteroidModel/"
            self.shape_url = modurl + "%s/shape.obj"%self.model_id
            self.spin_url = modurl + "%s/spin.txt"%self.model_id
            del ast, model, asts, models
            
        # download/read shape data file
        if self.shape_url is not None:
            try:
                shape_data = pd.read_csv(
                    self.shape_url, names=['type', '0', '1', '2'], delimiter='\s+'
                )
            except:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                shape_data = pd.read_csv(
                    self.shape_url, names=['type', '0', '1', '2'], delimiter='\s+'
                )
        elif shape_file is not None:
            shape_data = pd.read_csv(
                shape_file, names=['type', '0', '1', '2'], delimiter='\s+'
            )
        else:
            raise Exception("There's no `shape_url` or `shape_file`!")

        # parse shape data file; v lines are the x, y, z coords of the vertices
        vertices = shape_data[shape_data['type'] == 'v']
        self.nvertices = len(vertices)
        self.vertices = np.asarray(vertices[['0', '1', '2']], dtype=float)
        self.vertices_init = copy.copy(self.vertices)

        # f lines are the facets, defined by the indices (1-indexing) of the 3 vertices
        # that make up the triangular facet, clockwise for outside observer
        facets = shape_data[shape_data['type'] == 'f']
        self.nfacets = len(facets)
        self.facets = np.asarray(facets[['0', '1', '2']], dtype=int) - 1
        del shape_data, vertices, facets
        
        # get areas and unit normals for each facet by cross prods of 2 facet sides
        crosses = np.cross(
            self.vertices[self.facets][:,1] - self.vertices[self.facets][:,0],
            self.vertices[self.facets][:,2] - self.vertices[self.facets][:,0]
        )
        mags = np.linalg.norm(crosses, axis=1)
        self.areas = mags / 2
        self.norms = np.array([crosses[i] / mags[i] for i in range(self.nfacets)])
        self.norms_init = copy.copy(self.norms)
        del crosses, mags

        # calculate volume of the model by summing areas of each facet tetrahedron
        # vol tetrahedron = |a . (b x c)| / 6
        self.volume = np.sum(np.abs(np.diagonal(
            np.dot(
                self.vertices[self.facets][:,0,:], 
                np.cross(
                    self.vertices[self.facets][:,1,:],
                    self.vertices[self.facets][:,2,:]
                ).T
            )
        ))) / 6.
        self.diameter = (self.volume * 3/4/np.pi)**(1./3) * 2

        # apply units if the shape is calibrated
        if self.calibrated_size:
            self.vertices *= G3Units.km
            self.vertices_init *= G3Units.km
            self.diameter *= G3Units.km
            self.areas *= G3Units.km**2
            self.volume *= G3Units.km**3

        # download/read spin data file
        spin_data = None
        if self.spin_url is not None:
            import requests
            spin_data = requests.get(self.spin_url).text.split("\n")[:-1]
            spin_data = [line.split() for line in spin_data]
        elif spin_file is not None:
            with open(spin_file) as f:
                spin_data = f.readlines()
            spin_data = [line.split("\n")[0].split() for line in spin_data]
        
        # parse spin data file
        if spin_data is not None:
            self.spin_epoch = float(spin_data[1][0]) * G3Units.day # XXX jd handling
            self.spin_offset = float(spin_data[1][1]) * G3Units.deg
            self.spin_axis_init = np.array([0., 0., 1.])
            self.spin_long = float(spin_data[0][0]) * G3Units.deg
            self.spin_lat = float(spin_data[0][1]) * G3Units.deg
            self.period = float(spin_data[0][2]) * G3Units.hour
            self.spin_rate = 2 * np.pi / self.period
            del spin_data

    def calibrate_size(self, diameter=None, force_calibrate=False):
        """
        If the model is not already calibrated to real units, then it is scaled so that
        it has unit volume. Supply a diameter here (such that the volume of a sphere
        with that diameter has the same volume as this model) to calibrate the model.

        Arguments:
        ----------
        diameter : float [None]
            Volume-equivalent diameter in G3Units. If None provided, will try to query
            Small Body DataBase (SBDB) for a diameter to use.
        force_calibrate : bool [False]
            Whether to force calibration even if the model is already calibrated.
        """

        # raise an error if the size is already calibrated
        if self.calibrated_size and not force_calibrate:
            raise Exception("Trying to calibrate size when it's already calibrated!")

        # try to query SBDB for a diameter if one isn't provided
        if diameter is None:
            from astroquery.jplsbdb import SBDB
            props = SBDB.query(self.asteroid, phys=True, cache=False)
            diameter = props["phys_par"]["diameter"].value * G3Units.km
            del props

        # scale so model volume is equivalent to a sphere with requested diameter
        vol_sphere = 4. / 3. * np.pi * (diameter / 2.)**3
        lin_factor = (vol_sphere / self.volume) ** (1/3.)

        # save the scaling
        self.diameter = diameter
        self.vertices_init *= lin_factor
        self.vertices *= lin_factor
        self.areas *= lin_factor**2
        self.volume *= lin_factor**3
        self.calibrated_size = True

    def init_thermo(
        self,
        albedo=0.,
        eps=0.95,
        conductivity=2e-5 * G3Units.W / G3Units.cm / G3Units.K,
        thermal_inertia=50 * G3Units.W * G3Units.s**0.5 / G3Units.K / G3Units.m**2,
        depths=None,
        temp_distrib='sun_lat',
        reset_temps=True,
        verbose=False,
        fudge_factor = 1,
        a = 0
    ):
        """
        Sets various parameters for thermal modeling and checks if enough parameters are
        defined to start modeling.

        Arguments:
        ----------
        albedo : float [0.]
            Asteroid's bond albedo.
        eps : float [0.95]
            Asteroid's emissivity.
        conductivity : float [2e5 W/cm/K]
            Thermal conductivity in G3Units # do we need this???
        thermal_inertia : float [50 J/K/m^2/s^.5]
            Thermal inertia in G3Untis.
        depths : list-like [None]
            List of depths in G3Units below the subsurface to sample in the simulation.
            Will be stored in order of increasing depth. Default is same as used by
            Keihm 2013.
        temp_distrib : str ['sun_lat']
            Initial temperature distribution at a constant depth. Can be any of:
                'sun_lat' : max temp at subsolar, vary as cos(sun el - facet el)**.25
                'lat' : max temp at equat, vary as cos(facet el)
                'stm' : max temp at subsolar, vary as cos(facet pos from subsolar)**.25
                'const', None : constant temp over surface
            'sun_lat' is likely good for fast rotators, 'stm' likely good for slow
        reset_temps : bool [True]
            Whether to force setting the temperature distributions, even if they are
            already set.
        verbose : bool [False]
            Whether to print out some potentially helpful info.
        """

        # set provided variables
        self.albedo = float(albedo)
        self.eps = float(eps)
        self.cond = float(conductivity)
        self.therm_inert = float(thermal_inertia)
        self.Ls = self.cond / self.therm_inert / np.sqrt(self.spin_rate)

        # check that everything is set that needs to be
        if not self.calibrated_size:
            raise Exception("Model size not calibrate, run calibrate_size().")
        # XXX probably check or set things like albedo, etc here

        # set depths
        if depths is not None:
            raise NotImplementedError("Only use default depth and thicknesses for now.")
            self.depths = np.sort(depths)
        elif self.depths is None:
            #seasonal_thermal_wavelength = 20 # XXX this needs to not be hardcoded, also this is just a random number
            #self.depths = np.sort(np.concatenate([np.arange(0, 1, 0.2), np.geomspace(1, seasonal_thermal_wavelength, 20)])) * G3Units.mm
            ##self.depths = np.sort(np.concatenate([np.arange(0, 1, 0.2), np.geomspace(1, 1000, 20)])) * G3Units.mm    
            #self.depth = 60 * G3Units.cm # 100 * G3Units.cm
            self.depth = 48 *G3Units.cm #up to 40ish
            #self.depth =  8 * self.Ls
            self.thickness = 0.3 * G3Units.mm #0.3mm
            self.depths = np.arange(0., self.depth + self.thickness, self.thickness)
            self.thicknesses = np.ones_like(self.depths) * self.thickness
            if verbose:
                print("Setting depths [mm]:", self.depths / G3Units.mm)

        # set temperatures if not already set
        if (self.temps is None) or reset_temps:
            if temp_distrib not in [None, "sun_lat", "lat", "stm", "const"]:
                raise ValueError("Invalid initial temperature distribution requested.")

            # constant temperature scaling
            if temp_distrib in [None, 'const']:
                ang = np.zeros(self.nfacets) # we'll take cos of this, ie no scaling
                eta = 0.756 # same as STM, except every facet is as hot as subsolar

            # scale by facet norm elevation off from sun elevation
            elif temp_distrib == 'sun_lat':
                if self.sun is None:
                    raise Exception("Trying to set T based on sun without a sun location.")
                sun_el = xyz2rae(self.sun_corot)[-1]
                norms_el = xyz2rae(self.norms_init)[:,-1]
                ang = sun_el - norms_el
                eta = np.pi # XXX should eta also be pi here??

            # scale by facet norm elevation off from equator
            elif temp_distrib == "lat":
                ang = xyz2rae(self.norms_init)[:,-1]
                eta = np.pi

            # vary temp by facet azel from subsolar point, like Standard Thermal Model
            elif temp_distrib == "stm":
                sun_az, sun_el = xyz2rae(self.sun_corot)[1:]
                vertex_az, vertex_el = xyz2rae(self.vertices_init)[:,1:].T
                facet_mean_az = circmean(vertex_az[self.facets], axis=1)
                facet_mean_el = np.mean(vertex_el[self.facets], axis=1)
                ang = GreatCircleDistance(facet_mean_az, facet_mean_el, sun_az, sun_el)
                eta = 1.2 # just a guess based on some literature

            # calculate hottest temp based on energy balance at subsolar point in STM
            T = (self.solar_intensity / (G3Units.W/G3Units.m**2) * (1 - self.albedo) / sigma / self.eps / eta) ** 0.25 * G3Units.K
            if verbose:
                print(
                    "incident solar intensity [W/m^2]:",
                    self.solar_intensity / (G3Units.W/G3Units.m**2)
                )
                print("hottest temperature [K]:", T / G3Units.K)

            # calculate shape of temperature distribution
            distrib = np.max([np.cos(ang), np.zeros_like(ang)], axis=0) ** 00.25

            # scale chosen temp distrib by depth 
            self.temps = np.array([
                ( (1-a) * T * np.exp( - d / self.Ls / fudge_factor) + a * T ) * distrib for d in self.depths
            ])
        
    def set_sun_location(self, loc):
        """
        Sets the location of the sun.
        
        Arguments:
        ----------
        loc : np.array
            The x, y, z direction of the sun.
        """
        loc = np.array(loc, dtype=float)
        self.sun_dist = np.sqrt(np.sum(loc**2))
        self.sun = loc / self.sun_dist
        self.solar_intensity = L_sun.value / (4*np.pi*(self.sun_dist/G3Units.m)**2) * (G3Units.W/G3Units.m**2) # XXX constants

    def set_observer_location(self, loc):
        """
        Sets the location of the observer.
        
        Arguments:
        ----------
        loc : np.array
            The x, y, z direction of the observer.
        """
        loc = np.array(loc, dtype=float)
        self.observer_dist = np.sqrt(np.sum(loc**2))
        self.observer = loc / self.observer_dist
        
    def update_visibilities(self, area_eps=0.05):
        """
        Checks and updates what facets are sunlit and/or visible based on current sun
        and observer locations.

        Arguments:
        ----------
        area_eps : float [0.05]:
            Fraction of area of a facet that can be covered before it is considered
            overlapped by another surface.
        """
        if self.sun is not None:
            self.cos_sun, self.sunlit = self.check_visibility(
                direction=self.sun, ecliptic=True, area_eps=area_eps
            )
        if self.observer is not None:
            self.cos_obs, self.visible = self.check_visibility(
                direction=self.observer, ecliptic=True, area_eps=area_eps
            )

    def check_visibility(self, direction, ecliptic=True, area_eps=0.05, verbose=False):
        """
        Checks what facets are visible by an observer located far away in a specified
        direction. For convex shapes, this is equivalent to checking if each facet faces
        the observer.  For non-convex shapes, we check if there is overlap between
        facets when projected onto a plane perpendicular to the line of sight.

        Arguments:
        ----------
        direction : np.array
            The x, y, z unit direction of the observer.
        ecliptic : bool [True]
            Whether the direction is in ecliptic coords (True) or co-rot coords (False).
        area_eps : float [0.05]:
            Fraction of area covered before the facet is considered overlapped. Beware
            that setting to 0 may cause too many facets to get cut due to precisions.
        verbose : bool [False]
            Print out what facets shadow others in a non-convex shape, useful for
            bugfixing.
        
        Returns:
        --------
        cos_dir : np.array
            The cosine of the angle between each facet normal and the given dir
        visible : np.array
            A bool array of whether each facet is visible
        """

        # only sides facing in the selected direction might be visible
        cos_dir = np.dot(self.norms, direction)
        visible = np.array(cos_dir > 0., dtype=bool)

        # non-covex shapes may have overlapping facets, while convex shapes are easy
        if self.convex:
            return cos_dir, visible

        # rotate vertices so direction is parallel to z axis
        # rot by -a so vector lies over x axis then rot so it goes up to z axis
        r, a, e = xyz2rae(direction)
        R = np.dot(Ry(-(np.pi/2 - e)), Rz(-a)) 
        if ecliptic:
            vertices = np.dot(R, self.vertices.T).T
        else:
            vertices = np.dot(R, self.vertices_init.T).T

        # sort potentially vis facets by increasing closeness to observer
        vis_indices = np.array(range(len(visible)))[visible]
        z_coords = vertices[self.facets[vis_indices]][:,:,-1]
        vis_indices = vis_indices[np.argsort(np.mean(z_coords, axis=1))]

        # start with facets furthest from observer, go up until the closest facet
        if verbose:
            removed = 0
        for this_i in vis_indices[:-1]:

            # check if this facet overlaps with any of the facets closer to observer
            for closer_i in vis_indices[np.argwhere(vis_indices==this_i).item()+1:]:
                overlap = check_2Dtris_overlap(
                    t1 = vertices[self.facets[this_i]][:,:2],
                    t2 = vertices[self.facets[closer_i]][:,:2],
                    eps = area_eps * self.areas[this_i] * cos_dir[this_i],
                    allow_reversed = True,
                    on_boundary = False
                )

                # if this facet overlaps with any other closer, stop and remove it
                if overlap:
                    if verbose:
                        removed += 1
                        print("index {} overlapped by index {}".format(this_i, closer_i))
                    vis_indices = np.delete(vis_indices, np.where(vis_indices==this_i))
                    break

        # return the visible indices
        if verbose:
            print("removed n facets:", removed)
        visible = np.zeros_like(visible, dtype=bool)
        visible[vis_indices] = True
        return cos_dir, visible

    def set_locations(self, loc=None, sun_loc=None, earth_loc=None, update_vis=True): # XXX when storing values, maybe just have it so that it always calculates earth_loc, sun_loc values, just to not confuse people when looking at self values...
        """
        Calculates directions of Earth and Sun given ecliptic coordinate vectors of the
        asteroid, Earth, and Sun.  Only provide two of the arguments, depending on the
        origin of the given coordinates.

        If providing loc and sun_loc, assumes:
            loc : site-centric coords of asteroid
            sun_loc : site-centric coords of sun
        If providing loc and earth_loc, assumes:
            loc : heliocentric coords of asteroid
            earth_loc : heliocentric coords of earth
        If providing sun_loc and earth_loc, assumes:
            sun_loc : heliocentric coords of asteroid
            earth_loc : site-centric coords of asteroid

        Arguments:
        ----------
        loc, sun_loc, earth_loc : np.array
            x, y, z coordinates in the ecliptic ref frame
        update_vis : bool [True]
            Whether to also update the sunlit/visible facets (True) or just set the
            locations of the sun and earth (False).
        """

        if loc is not None and sun_loc is not None and earth_loc is not None:
            raise Exception("Cannot parse origin for location arguments.")

        # assumes heliocentric coords
        if loc is not None and earth_loc is not None: 
            # store values
            self.loc = np.array(loc)
            self.earth_loc = np.array(earth_loc)

            # update unit directions and distances
            self.set_observer_location(self.earth_loc - self.loc)
            self.set_sun_location(-self.loc)
        
        # assumes geocentric coords
        elif loc is not None and sun_loc is not None:
            # store values
            self.loc = np.array(loc)
            self.sun_loc = np.array(sun_loc)

            # update unit directions and distances
            self.set_observer_location(-self.loc)
            self.set_sun_location(self.sun_loc - self.loc)

        # assumed geocentric obj xyz for earth loc, and heliocentric obj xyz for sun loc
        elif earth_loc is not None and sun_loc is not None:
            # store values
            self.earth_loc = np.array(earth_loc)
            self.sun_loc = np.array(sun_loc)

            # update unit directions and distances
            self.set_observer_location(-self.earth_loc)
            self.set_sun_location(-self.sun_loc)

        # update visible/sunlit facets if requested
        if update_vis:
            self.update_visibilities()

    def rotate(self, epoch=None, time_delta=None):
        """
        Rotates the asteroid by updating vertices and norms to where they point at the
        specified epoch.
        
        Arguments:
        ----------
        epoch : float or G3Time [None]
            The epoch in JD you want the orientation at. Provide epoch or time_delta # XXX jd handling
        time_delta : float [None]
            The amount to increase the current epoch by. Provide epoch or time_delta
        """
        # calculate new epoch
        if epoch is not None and time_delta is not None:
            raise Exception("Cannot rotate with an epoch and a time_delta")
        elif time_delta is not None:
            if self.epoch is None:
                epoch = self.spin_epoch + time_delta
            else:
                epoch = self.epoch + time_delta

        # calculate the rotation matrix
        R = Rz(self.spin_offset + self.spin_rate*(epoch - self.spin_epoch))
        R = np.dot(Ry(np.pi/2 - self.spin_lat), R)
        R = np.dot(Rz(self.spin_long), R)

        # rotate vectors from initial reference position
        self.vertices = np.dot(R, self.vertices_init.T).T
        self.norms = np.dot(R, self.norms_init.T).T
        self.spin_axis = np.dot(R, self.spin_axis_init)
        self.x_axis = np.dot(R, [1., 0., 0.])
        self.y_axis = np.dot(R, [0., 1., 0.])
        self.epoch = epoch

        # update visible and sunlit facets
        self.update_visibilities()

        # unrotate sun and observer vectors
        self.sun_corot = np.dot(np.linalg.inv(R), self.sun)
        self.observer_corot = np.dot(np.linalg.inv(R), self.observer)

    def update_all_to_epoch(
        self, epoch=None, time_delta=None, ephem_file=None, site="SPT"
    ):
        """
        Given an epoch, rotates the asteroid and updates locations and orientations.
        
        Arguments:
        ----------
        epoch : float [None]
            The epoch in JD you want the orientation at. Provide epoch or time_delta # XXX jd handling
        time_delta : float [None]
            The amount you want to increase the current epoch by. Provide epoch or
            time_delta
        ephem_file : str [None]
            Path and filename of ephemeride file to read in. If not provided, will
            attempt to query JPL Horizons for info; this is not recommended because it
            is much slower.
        site : str ['SPT']
            The location you're observing from, needed if no ephem_file provided.
        """
        # calculate new epoch
        if epoch is not None and time_delta is not None:
            raise Exception("Cannot update with both an epoch and a time_delta")
        elif time_delta is None:
            if self.epoch is None:
                time_delta = epoch - self.spin_epoch
            else:
                time_delta = epoch - self.epoch
        elif time_delta is not None:
            if self.epoch is None:
                epoch = self.spin_epoch + time_delta
            else:
                epoch = self.epoch + time_delta
        self.epoch = epoch

        # read in ephemerides if provided
        if ephem_file is not None:
            frame = list(core.G3File(ephem_file))[0] # XXX be more robust than this
            times = np.array([t.mjd + 2400000.5 for t in frame["Ephem"].times]) * G3Units.day # XXX jd handling
            if epoch < min(times) or epoch > max(times):
                raise Exception(
                    "epoch out of range of ephem file times ({} - {})".format(
                        min(times) / G3Units.day, max(times) / G3Units.day
                    )
                )
            self.set_locations(
                sun_loc=[
                    np.interp(x=epoch, xp=times, fp=frame["Ephem"]["HeliocentricEclipticX"]),
                    np.interp(x=epoch, xp=times, fp=frame["Ephem"]["HeliocentricEclipticY"]),
                    np.interp(x=epoch, xp=times, fp=frame["Ephem"]["HeliocentricEclipticZ"])
                ],
                earth_loc=[
                    np.interp(x=epoch, xp=times, fp=frame["Ephem"]["GeocentricEclipticX"]),
                    np.interp(x=epoch, xp=times, fp=frame["Ephem"]["GeocentricEclipticY"]),
                    np.interp(x=epoch, xp=times, fp=frame["Ephem"]["GeocentricEclipticZ"])
                ],
                update_vis=False
            )

        # otherwise query for info (slow and not preferrable)
        else:
            from astroquery.jplhorizons import Horizons
            e = Horizons(
                id=self.asteroid, location=site, epochs=epoch/G3Units.day, id_type="smallbody"
            ).vectors(refplane="ecliptic", cache=False) # XXX jd date handling
            s = Horizons(
                id=self.asteroid, location="@10", epochs=epoch/G3Units.day, id_type="smallbody"
            ).vectors(refplane="ecliptic", cache=False) # XXX jd date handling
            self.set_locations(
                sun_loc=np.array([s["x"][0], s["y"][0], s["z"][0]]) * G3Units.AU,
                earth_loc=np.array([e["x"][0], e["y"][0], e["z"][0]]) * G3Units.AU,
                update_vis=False
            )
        
        # update epoch to account for light delay while traveling to observer
        # we should re-query pos after, but this is a minor error that saves time
        self.epoch = self.epoch - self.observer_dist / (c * G3Units.m/G3Units.s)

        # rotate the asteroid
        self.rotate(epoch=self.epoch)

    def get_subsolar_index(self):
        """
        Returns the index of the facet closest to the subsolar point.

        Returns:
        ----------
        int
            Index of the subsolar facet.
        """
        sun_az, sun_el = xyz2rae(self.sun_corot)[1:]
        vertex_az, vertex_el = xyz2rae(self.vertices_init)[:,1:].T
        facet_mean_az = circmean(vertex_az[self.facets], axis=1)
        facet_mean_el = np.mean(vertex_el[self.facets], axis=1)
        ang = GreatCircleDistance(facet_mean_az, facet_mean_el, sun_az, sun_el)
        return np.argsort(ang)[0]

    def precalculate_sunlit(self, step_size=1*G3Units.deg, n_steps=None):
        """
        Given current orbit geometry, fully rotate the asteroid by small steps and 
        calculate sunlit facets at each step. For non-convex shapes, this saves
        significant processing time while equilibrating temperatures if multiple
        rotations are required.

        Arguments:
        ----------
        step_size : float [1 * G3Units.deg]
            Angle size in G3Units of each step. Provide this or n_steps.
        n_steps : int [None]
            Number of steps to take in the rotation. Provide this or step_size. For
            instance, setting n_steps=360 is equivalent to step_size=1*G3Units.deg.
        """
        # format input parameters
        time_step = self.period / n_steps #set n as number of steps to take per rotation
        init_epoch = self.epoch
        facet_stat=[]  
        facet_stat.append(self.sunlit * self.cos_sun)
        for i in range(n_steps): # iterate through each orientation
            self.rotate(time_delta= time_step) # rotate the asteroid to that time_delta
            facet_stat.append(self.sunlit*self.cos_sun) # calculate sunlit facets
            # store calculated info
        self.precalculated_cos_sun = np.array(facet_stat)
        self.precalculated_time_step = time_step
        self.precalculated_n_steps = n_steps
        self.epoch = init_epoch
        
    def update_temps_one_rotation(self, n_steps=None, facet=None): 
        """
        Given current orbit geometry, fully rotate the asteroid by small steps and, 
        iterating through the list of facets, update the temperature of each facet. 

        Arguments:
        ----------
        n_steps : int [None]
            Number of steps to take in the rotation. Provide this or step_size. For
            instance, setting n_steps=360 is equivalent to step_size=1*G3Units.deg.
        facet : int [None]
            Index of a specific facet. For instance, to use the subsolar facet, 
            set facet=self.get_sublolar_index().
        """
        #check to see if precalculate_sunlit was run
        if self.precalculated_time_step is None or self.precalculated_n_steps != n_steps:
                self.precalculate_sunlit(n_steps=n_steps) 
        if facet is None: #default facet setting
            #iterate through each facet
            list_of_facets = range(self.nfacets) 
        else:
            if np.isscalar(facet): #setting for specific facet
                facet = [facet] #converts to a list
            list_of_facets= facet  
        
        #iterate through list_of_facets and update temp of each for one rotation
        for facet_index in list_of_facets:  
            t, x, temp_array = thermo.heat_equation_finite_difference(time_final = self.period , 
                                                  dtime = self.precalculated_time_step ,
                                                  L = self.depth ,
                                                  nx = len(self.depths) ,
                                                  initial_temp = self.temps[:, facet_index] ,
                                                  conductivity = self.cond ,
                                                  heat_capacity = self.therm_inert**2 / self.cond ,
                                                  density = 1 ,
                                                  emissivity = self.eps ,
                                                  bc_type = [3, 1] ,
                                                  bc = [None, 0] ,
                                                  incident_flux = [self.solar_intensity * self.precalculated_cos_sun[:, facet_index],0]
                                                  )
            #update and store new self.temps array
            self.temps[:,facet_index] = copy.copy(temp_array[-1,:])

    def fit_full_rotation(self, facet = None): 
         """
        Rotates the asteroid twice and uses the least squares fitting tool to
        optimize the fit of the initial temperature guess function, then updates
        the previous self.temps to the fitted temperature guess, and repeats the
        double rotation and fitting process. Finally rotates the asteroid four
        times and updates the temperature of each visible facet.

        Arguments:
        ----------
        facet : int [None]
            Index of specific facet. If set to None, will iterate through 
            a list of all facets.
       
        """
        #define a function for fitting 
        def temperature_guess(p, depths): #p = the three parameters: temp, fudge1, and fudge2
            return (1-p[2]) * p[0] * np.exp( - depths / self.Ls / p[1]) + p[2] * p[0] #fitting function 

        for j,facet_index in enumerate(np.arange(self.nfacets)[self.visible]):
            print('%s/%s'%(j,np.sum(self.visible)))
            for i in range(2):
                #rotate the asteroid and update the temps with the new guess
                self.update_temps_one_rotation(n_steps = 360, facet = facet_index )
            
            #use least squares function to optimize fit 
            try:
                fit = least_squares(
                    fun=lambda p : temperature_guess(p, depths=self.depths[:100]) - self.temps[:100, facet_index],
                    x0=[200*core.G3Units.K,1,.8],
                    bounds=([0,0,0],[np.inf,5,1])
                    ) 
                #set previous guess equal to new guess
                self.temps[:,facet_index] = temperature_guess(fit['x'], depths=self.depths)
            except:
                self.temps[:,facet_index] = 0
                continue

            for i in range(2):
                #rotate the asteroid and update the temps with the new guess
                self.update_temps_one_rotation(n_steps = 360, facet = facet_index )

            #use least squares function to optimize fit 
            try:
                fit = least_squares(
                    fun=lambda p : temperature_guess(p, depths=self.depths[:100]) - self.temps[:100, facet_index],
                    x0=[200*core.G3Units.K,1,.8],
                    bounds=([0,0,0],[np.inf,5,1])
                    ) 
                #set previous guess equal to new guess
                self.temps[:,facet_index] = temperature_guess(fit['x'], depths=self.depths)
            except:
                self.temps[:,facet_index] = 0
                continue
            
            #rotate the asteroid and update facet temperatures for all visible facets
            for i in range(4):
                self.update_temps_one_rotation(n_steps = 360, facet = facet_index)
           
        

# ==================================================================================
# shape model functions
# ==================================================================================

def update_temps(ast, time_delta=None):
    """
    Solve the one-dimensional transient heat conduction equation numerically with
    implicit finite difference methods given current sun orientation and temperature
    distribution. Updates from previous time up to some time delta.

    Heat equation has with outgoing thermal emission at all layers, zero flux BC for the
    bottom node (ideally ~5x the seasonal thermal wavelength), and incident solar flux
    at the outer layers. 
    """
    # do math
    # update ast.temps
    return

def equilibrate_temp(ast, eps=None, n_iter=None, n_steps=None):
    """
    Given current solar orientation and temperature distribution, rotate asteroid and
    solve transient heat conduction until some condition is met. Some ideas:
        - temperatures change by less than some epsilon percentage
        - do n_iterations of rotations (with n_steps per iteration)
    """
    # calculate size of time_delta to evaluate heat eqn, based on input or
    # defaults

    # call update_temps(time_delta) either for requested number of iterations or
    # until requested convergence

    return

def flux(ast, freq):
    """
    Calculates and returns the surface-integrated flux density for the current viewing
    geometry and temperature distribution at the requested frequency.

    See the following for more details on calculations:
        Keihm et al. 2013 -- eqns 1, 2
        Lagerros 1996b

    Arguments:
    ----------
    ast : AsteroidShapeModel
        Loaded asteroid model with a defined temperature profile.
    freq : float
        Frequency in G3Units to calculate the flux density at.

    Returns:
    --------
    F : flaot
        Observable flux density in G3Units.
    """

    # make sure the model has a physical size, otherwise the output is meaningless
    if not ast.calibrated_size:
        raise Exception(
            "Size isn't calibrated. Use a calibrated model or run calibrate_size()."
        )

    # electrical skin depth estimates, Keihm+2013 citing Gary and Keihm 1978
    wavelength = (c * G3Units.m / G3Units.s ) / freq
    if wavelength <= 1 * G3Units.mm:
        Le = 7 * wavelength
    elif wavelength >= 2 * G3Units.cm:
        Le = (1.2/G3Units.cm * wavelength + 6.9) * wavelength
    else:
        Le = (2.1/G3Units.cm * wavelength + 6.8) * wavelength
    print('wavelength [mm]:', wavelength / G3Units.mm)
    print('Le:', Le / wavelength, '* wavelengths')
    
    # calculate the flux coming from each depth
    F = 0
    for z, dz, T in zip(ast.depths, ast.thicknesses, ast.temps):

        # only consider flux from visible facets
        T = T[ast.visible]
        cos_obs = ast.cos_obs[ast.visible]
        dA = ast.areas[ast.visible] * cos_obs

        # Planck flux from each facet
        B = 2 * h * (freq/G3Units.Hz)**3 / c**2 / (np.exp(h * (freq/G3Units.Hz) / k / (T/G3Units.K)) - 1.) # XXX constants
        B *= G3Units.W / G3Units.m**2 / G3Units.Hz

        # integrate over facets, attenuated by layers above current depth
        #integrand = B
        integrand = B * np.exp(- z / Le / cos_obs) / Le / cos_obs
        #integrand = B * np.exp(- z / self.Le / cos_obs) / self.Le / cos_obs
        F += np.sum(integrand * dA * dz)

    return ast.eps * F / ast.observer_dist**2
    #return 1 / np.pi / ast.observer_dist**2 * ast.eps * F # XXX Keihm13 says eps -> 1-Rv, check this...
    # XXX I think this pi discrepancy may be from definition of Planck's funct. not including it matches my other STM, NEATM, etc math
    
def print_details(ast):
    """
    Prints out some projection details at current viewing geometry. This is useful for
    debugging, especially compared to projection details while viewing model on DAMIT.

    Arguments:
    ----------
    ast : AsteroidShapeModel
        Loaded asteroid model updated to an epoch's viewing geometry.
    """

    # subsolar info
    sun_rae = xyz2rae(ast.sun_corot)
    print(
        "Subsolar-point longitude (°):",
        np.round(np.mod(sun_rae[1]/G3Units.deg,360), 1)
    )
    print("Subsolar-point latitude (°):", np.round(sun_rae[2]/G3Units.deg, 1))

    # observer info
    obs_rae = xyz2rae(ast.observer_corot)
    print(
        "Subearth-point longitude (°):",
        np.round(np.mod(obs_rae[1]/G3Units.deg,360), 1)
    )
    print("Subearth-point latitude (°):", np.round(obs_rae[2]/G3Units.deg, 1))

    # phase-angle bisector info
    bisector = xyz2rae(ast.observer_corot + ast.sun_corot)
    print(
        "Phase-angle-bisector longitude (°):",
        np.round(np.mod(bisector[1]/G3Units.deg,360), 1)
    )
    print("Phase-angle-bisector latitude (°):", np.round(bisector[2]/G3Units.deg, 1))
    print(
        "Solar phase angle (°):",
        np.round(np.arccos(np.dot(ast.observer, ast.sun))/G3Units.deg, 1)
    )


def plot(
    ast, arrows=["sun", "obs", "spin"], plot_rotated=True, highlight_sunlit=False,
    highlight_visible=False, cmap="Greys_r", plot_these_ind=[], filename=None,
    show_plot=True, orientation=None, legend=False
):
    """
    Plots the asteroid shape model.
    
    Arguments:
    ----------
    ast : AsteroidShapeModel
        Loaded asteroid model.
    arrows : listlike ["sun", "obs", "spin"]
        Which arrows to plot out of "sun" (dir to sun as yellow arrow), "obs" (dir to
        observer as a blue arrow), and "spin" (direction of rotational axis as green
        arrow). Provide empty list to plot no arrows.
    plot_rotated : bool [True]
        Whether to plot in rotated (True) or in co-rotating frame (False)
    highlight_sunlit : bool [False]
        Whether to show sunlit facets in a different color.
    highlight_visible : bool [False]
        Whether to show visible facets in a different color.
    plot_these_ind : list-like [[]]
        If provided, plots only the requested facets. Useful for bugfixing.
    cmap : str ["Greys_r"]
        Colormap to plot the facets in.
    filename : str [None]
        If provided, saves the output image to path filename.
    show_plot : bool [True]
        Whether or not to show the final plot.
    orientation : str or tuple [None]
        If a string is provided, whether you want the "observer", "sun", or "spin"
        vectors oriented towards the viewer in the output plot. If a tuple is provided,
        the (az, el) you want oriented towards the viewer.  Default is matplotlib's
        default isometric view. "oberserver_r", etc., plots so that that vector is
        oriented away from the viewer.
    legend : Bool [False]
        Whether to plot the legend
    """
    import mpl_toolkits.mplot3d as a3
    import matplotlib.pyplot as plt
    import matplotlib.colors as mpc
    import matplotlib.tri as mtri

    fig = plt.figure()
    ax = plt.subplot(projection="3d")

    # select and scale vertices and norms for plotting
    vertices = ast.vertices if plot_rotated else ast.vertices_init
    norms = ast.norms if plot_rotated else ast.norms_init
    if ast.calibrated_size:
        vertices = vertices / G3Units.km

    # set up light source
    if ast.sun is not None:
        sun = ast.sun if plot_rotated else ast.sun_corot
        ls = mpc.LightSource(
            azdeg=np.arctan2(sun[0], sun[1]) / G3Units.deg,
            altdeg=90. - np.arccos(sun[2]) / G3Units.deg
        )
    else:
        ls = mpc.LightSource(azdeg=0, altdeg=90)

    # plot triangle facets
    if len(plot_these_ind):
        triang = mtri.Triangulation(vertices[:,0], vertices[:,1], triangles=ast.facets[plot_these_ind])
        ax.plot_trisurf(triang, vertices[:,2], color='w', lightsource=ls)
    elif highlight_sunlit and hightlight_visible:
        raise Exception("Can't plot visible and sunlit facets!")
    elif highlight_sunlit or hightlight_visible:
        which = ast.sunlit if highlight_sunlit else ast.visible
        c = "orange" if highlight_sunlit else "cornflowerblue"
        triang = mtri.Triangulation(vertices[:,0], vertices[:,1], triangles=ast.facets[~which])
        ax.plot_trisurf(triang, vertices[:,2], color="w", lightsource=ls)
        triang = mtri.Triangulation(vertices[:,0], vertices[:,1], triangles=ast.facets[which])
        ax.plot_trisurf(triang, vertices[:,2], color=c, lightsource=ls)
    else:
        triang = mtri.Triangulation(vertices[:,0], vertices[:,1], triangles=ast.facets)
        ax.plot_trisurf(triang, vertices[:,2], color="w", lightsource=ls)
    
    # plot sun direction
    mag = np.sqrt(np.max(np.sum(np.square(vertices), axis=1)))
    if "sun" in arrows and ast.sun is not None:
        sun = ast.sun if plot_rotated else ast.sun_corot
        ax.quiver(
            sun[0]*mag, sun[1]*mag, sun[2]*mag,
            sun[0]*mag*1.3, sun[1]*mag*1.3, sun[2]*mag*1.3,
            color="orange", label="Sun"
        )
    
    # plot observer direction
    if "obs" in arrows and ast.observer is not None:
        obs = ast.observer if plot_rotated else ast.observer_corot
        ax.quiver(
            obs[0]*mag, obs[1]*mag, obs[2]*mag,
            obs[0]*mag*1.3, obs[1]*mag*1.3, obs[2]*mag*1.3,
            color="steelblue", label="Observer"
        )
    
    # plot rotational axis
    if "spin" in arrows:
        spin = ast.spin_axis if plot_rotated else ast.spin_axis_init
        ax.quiver(
            spin[0]*mag, spin[1]*mag, spin[2]*mag,
            spin[0]*mag*1.3, spin[1]*mag*1.3, spin[2]*mag*1.3,
            color="darkgreen", label="Rotation Axis"
        )
    
    # plot vertices
    if plot_vertices:
        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c="k", s=0.5)
    else:
        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c="k", s=0.5, alpha=0.)
    
    # labels
    label = "%s"
    if ast.calibrated_size:
        label = label + " [km]"
    if not plot_rotated:
        label = "Co-Rotating " + label
    ax.set_xlabel(label%"X")
    ax.set_ylabel(label%"Y")
    ax.set_zlabel(label%"Z")
    if legend:
        ax.legend()

    # kleuged way of fixing the aspect ratio, since mpl won't do that for 3d...
    X = np.array([1, 1, 1, 1, -1, -1, -1, -1])*mag*1.5
    Y = np.array([1, 1, -1, -1, 1, 1, -1, -1])*mag*1.5
    Z = np.array([1, -1, 1, -1, 1, -1, 1, -1])*mag*1.5/1.33
    for x, y, z in zip(X, Y, Z):
        ax.plot(x, y, z, "w")

    # rotate to requested viewing orientation
    def set_orientation(xyz):
        az = np.arctan2(xyz[1], xyz[0]) / G3Units.deg
        el = 90. - np.arccos(xyz[2]) / G3Units.deg
        # roll = np.arccos(self.spin_axis[2]) / deg # doesnt work for matplotlib 3.4.0
        ax.view_init(azim=az, elev=el)
    if orientation is None:
        pass
    elif ast.observer is not None and orientation=="observer":
        set_orientation(ast.observer)
    elif ast.observer is not None and orientation=="observer_r":
        set_orientation(-ast.observer)
    elif ast.sun is not None and orientation=="sun":
        set_orientation(ast.sun)
    elif ast.sun is not None and orientation=="sun_r":
        set_orientation(-ast.sun)
    elif ast.spin_axis is not None and orientation=="spin":
        set_orientation(ast.spin_axis)
    elif ast.spin_axis is not None and orientation=="spin_r":
        set_orientation(-ast.spin_axis)
    elif not np.isscalar(orientation):
        ax.view_init(azim=orientation[0], elev=orientation[1])

    # outputs
    if filename is not None:
        plt.savefig(filename, dpi=300)
        if not show_plot:
            plt.close(fig) # not closing clogs up memory when making many plots
    if show_plot:
        plt.show()

def plot_2d(
    ast, which="temps", messyfast=True, naz=30, nel=20, filename=None,
    show_plot=True, cmap="coolwarm", depth=0, title='', facets_to_plot="all",
    vmin=None, vmax=None
):
    """
    Plots the temperature distribution over the shape model in 2D.
    
    Arguments:
    ----------
    ast : AsteroidShapeModel
        Loaded asteroid model.
    which : str ['temps']
        What to plot out of: 'temps', 'cos_sun', 'cos_obs'
    messyfast : bool [True]
        Plots each facet quickly, but facets near poles are plotted incorrectly.
        If False, makes an az el grid and slowly plots temp of facets at grid
        azel points. 
    naz, nel : floats [30, 20]
        Number of azimuth, elivation points in the plot grid when plotting with
        messyfast False
    filename : str [None]
        If provided, saves the output image to path filename.
    show_plot : bool [True]
        Whether or not to show the final plot.
    cmap : str ["coolwarm"]
        Colormap to plot the facets in.  Recommended 'coolwarm' or 'inferno'
    depth : int [0]
        If plotting 'temps', which depth layer index to plot.
    title : str [""]
        Plot title.
    facets_to_plot : str ["all"]
        Which facets to plot when messyfast is true: "all", "visible", "sunlit"
    vmin, vmax : float [None]
        Min and max color scales to plot. Default chooses min and max of data.
    """
    import matplotlib.pyplot as plt 

    if ast.temps is None:
        ast.temps = ast.cos_sun

    # initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="mollweide")

    # choose what to plot
    if which == "temps":
        to_plot = ast.temps[depth] / G3Units.K
        cbar = "Temperature at Depth {}cm [K]".format(
            np.round(ast.depths[depth] / G3Units.cm, 2)
        )
        print(np.min(to_plot), np.max(to_plot))
    elif which == "cos_sun":
        to_plot = ast.cos_sun
        cbar = "Cos of Facet Normal with Sun Direction"
    elif which == "cos_obs":
        to_plot = ast.cos_obs
        cbar = "Cos of Facet Normal with Earth Direction"
        
    # choose color scale
    if vmin is None:
        vmin = np.nanmin(to_plot)
    if vmax is None:
        vmax = np.nanmax(to_plot)

    # this gets close, but triangles near the poles are messed up...
    if messyfast:
        
        # pick out the selected facets
        which_facets = []
        if facets_to_plot == "all":
            which_facets = np.arange(ast.nfacets)
        elif facets_to_plot == "visible":
            which_facets = np.arange(ast.nfacets)[ast.visible]
        elif facets_to_plot == "sunlit":
            which_facets = np.arange(ast.nfacets)[ast.sunlit]
        
        for i in which_facets:
            rae = xyz2rae(ast.vertices_init[ast.facets[i]])
            az = rae[:, 1]
            el = rae[:, 2]
            color = get_color(
                val=to_plot[i], vmin=vmin, vmax=vmax,
                cmap=cmap
            )
            if np.any(np.abs(az[1:] - az[:-1]) > 90 * G3Units.deg):
                where = az < 0
                az[where] += 360 * G3Units.deg
                ax.fill(az, el, c=color, alpha=0.7)
                az -= 360 * G3Units.deg
            ax.fill(az, el, c=color, alpha=0.7)

        import matplotlib.colors as mpc
        from matplotlib import cm
        color_norm = mpc.Normalize(vmin=vmin, vmax=vmax)
        im = cm.ScalarMappable(norm=color_norm, cmap=plt.get_cmap(cmap))

    else:
        # make a grid of az, el and temps to plot
        az, el = np.meshgrid(
            np.linspace(-np.pi, np.pi, naz),
            np.linspace(-np.pi/2, np.pi/2, nel),
            indexing="xy"
        )
        color = np.zeros_like(az)

        # go through each az, el point
        
        for i in range(naz):
            for j in range(nel):

                # find the facet that covers that az, el
                # start at previous k to save time for nearby azel
                this_K = 0
                for K in np.roll(range(ast.nfacets), -this_K):
                    intercepts, where = ray_triangle_intersection(
                        ray_start=np.array([0.,0.,0.]),
                        ray_vec=rae2xyz([1., az[j,i], el[j,i]]),
                        triangle=ast.vertices_init[ast.facets[K]]
                    )
                    if intercepts:

                        # save color of that facet and move on to next az, el
                        color[j,i] = to_plot[K]
                        this_K = K
                        break

        # plot the figure
        im = ax.pcolormesh(az, el, color, cmap=cmap, vmin=vmin, vmax=vmax)

    sun_lon, sun_lat = xyz2rae(ast.sun_corot)[1:]
    ax.scatter(x=sun_lon, y=sun_lat, marker="o", c="orange", label="Subsolar-point")
    obs_lon, obs_lat = xyz2rae(ast.observer_corot)[1:]
    ax.scatter(x=obs_lon, y=obs_lat, marker="P", c="darkgreen", label="Subearth-point")
    ax.legend(bbox_to_anchor=(0,0))

    plt.colorbar(im, label=cbar)
    plt.grid(alpha=0.5, c="white")
    ax.set_ylabel("Co-Rotating Latitude [deg]")
    ax.set_xlabel("Co-Rotating Longitude [deg]")
    ax.legend(bbox_to_anchor=(0,0))
    ax.set_title(title)

    # outputs
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        if not show_plot:
            plt.close(fig)
    if show_plot:
        plt.show()
    
def plot_gif(ast,n, frame_length=0.2): 
    """
    Creates a GIF of a full rotation of the asteroid shape model.
    
    Arguments:
    ----------
    ast : AsteroidShapeModel
        Loaded asteroid model.
    n : int
        Number of steps taken in a full rotation.
    frame_length : float [0.2]
        Length of each frame in rotation shown in GIF.
    """
    import imageio  #to make GIF
    import os  #to delete pngs after GIF is made
    
    #make n pngs (number of steps taken in one rotation)
    time_step = ast.period / n
    for i in range(n):
        ast.rotate(time_delta= time_step)
        asm.plot(
            ast,
            filename='temporaryrotation.'+ str(i) + '.png',
            show_plot=False
        )
    files = np.sort(glob.glob('temporaryrotation.*.png'))  #sort the png files in chronological order
    fileno = []
    for file in files:
        fileno.append(int(file.split('.')[1]))

    files = files[np.argsort(fileno)]

    images = [] #create list of each image
    for file in files:
        images.append(imageio.imread(file)) #read in png files

    imageio.mimsave('rotation.gif', images, 'GIF', duration=frame_length)  #create and save GIF of images
    
    # delete the intermediate pngs
    for file in files:
        os.remove(file)  #saves space