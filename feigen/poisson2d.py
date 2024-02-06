import numpy as np
import splinepy
import vedo
from scipy.sparse import csr_array, linalg

from feigen._base import FeigenBase

# skip bound checks
splinepy.settings.CHECK_BOUNDS = False
splinepy.settings.NTHREADS = 2

_CONF = {
    "sphere_option": {
        "res": 8,
        "r": 0.04,
        "c": "red",
    }
}


class _BoundaryConnection:
    """
    Helper class to transform coupled boundary points with 2 steps:
    1. Get coupled boundary id and control point id thereof
    2. Compute relative movement

    Parameters
    ----------
    boundary_splines: list

    """

    __slots__ = (
        "splines",
        "end_ids",
        "front_connection",
        "end_connection",
        "connection",
        "direction_factors",
        "corners",
    )

    def __init__(self, boundary_splines=None):
        if boundary_splines is not None:
            self.setup(boundary_splines)

    def setup(self, boundary_splines):
        # this is for 2D splines so we can hard code this
        self.splines = boundary_splines

        # get id of connected boundary
        # (begin, end)
        self.connection = (
            (2, 3),
            (2, 3),
            (0, 1),
            (0, 1),
        )

        # id of end cps, they repeat.
        self.end_ids = (
            int(len(boundary_splines[0].cps) - 1),
            int(len(boundary_splines[0].cps) - 1),
            int(len(boundary_splines[2].cps) - 1),
            int(len(boundary_splines[2].cps) - 1),
        )

        # this is a bit verbose and can be also done with mod but
        # this way, we can do a direct look up
        # plug in from_bid
        self.front_connection = (0, self.end_ids[2], 0, self.end_ids[0])
        self.end_connection = (0, self.end_ids[3], 0, self.end_ids[1])

        # some coupled boundaries flip directions to visualize
        # positive magnitudes. these are factors one can multiply to
        # change directions per axis
        # swapping axis is done separately with fixed permutation [1, 0, 2]
        self.direction_factors = (
            (np.array([1, 1]), np.array([-1, 1])),
            (np.array([-1, 1]), np.array([1, 1])),
            (np.array([1, 1]), np.array([1, -1])),
            (np.array([1, -1]), np.array([1, 1])),
        )

        # initial corners - used to compute differences
        # again, (begin, end)
        self.corners = (
            (self.splines[0].cps[0].copy(), self.splines[0].cps[-1].copy()),
            (self.splines[1].cps[0].copy(), self.splines[1].cps[-1].copy()),
            (self.splines[2].cps[0].copy(), self.splines[2].cps[-1].copy()),
            (self.splines[3].cps[0].copy(), self.splines[3].cps[-1].copy()),
        )

    def transform_ids(self, bid, cp_id):
        if cp_id == 0:
            return self.connection[bid][0], self.front_connection[bid]

        if cp_id == self.end_ids[bid]:
            return self.connection[bid][1], self.end_connection[bid]

        return None, None

    def transform_position(  # noqa PLR0913
        self, to_bid, to_cp_id, from_bid, from_cp_id, from_position
    ):
        """
        This call assumes that from_cp_id is at either end of the boundary
        spline.
        """
        # get lookup id
        which_end = 1 if from_cp_id != 0 else 0
        to_which_end = 1 if to_cp_id != 0 else 0

        # compute the difference - this flips axis and apply correct direction
        difference = (from_position - self.corners[from_bid][which_end])[
            [1, 0]
        ] * self.direction_factors[to_bid][to_which_end]

        return self.corners[to_bid][to_which_end] + difference


def _process_spline_actors(plt):
    """
    Helper function to convert current splines to showables
    (spline_actors).

    Parameters
    ----------
    plt: BSpline2D

    Returns
    -------
    None
    """
    # set show_options
    spl = plt._s["spline"]
    show_options = spl.show_options
    show_options["resolutions"] = plt._c["sample_resolutions"]
    show_options["control_points"] = False
    show_options["control_mesh"] = True
    show_options["lighting"] = "off"

    # get spline actors
    plt._s["spline_actors"] = spl.showable()

    # don't want any of these actors to be pickable
    for v in plt._s["spline_actors"].values():
        v.pickable(False)

    # process cps
    # nothing selected -> update all
    cp_id = plt._s["picked_cp_id"]
    new_cps = []
    if cp_id < 0:
        for i, cp in enumerate(plt._s["spline"].cps):
            sph = vedo.Sphere(cp, **_CONF["sphere_option"])
            sph.cp_id = i
            new_cps.append(sph)
        plt._s["spline_cp_actors"] = new_cps
        return None

    sphere_mesh = plt._s["spline_cp_actors"][cp_id]
    sphere_mesh.pos(*plt._s["spline"].cps[cp_id])
    sphere_mesh.apply_transform_from_actor()


def _process_boundary_actors(plt):
    """
    Helper function to process boundary spline.
    all necessary values should be available from _s

    Parameters
    ----------
    plt: BSpline2D

    Returns
    -------
    None
    """
    # get boundary_spline
    bid = plt._s["picked_boundary_id"]

    bids = []

    if bid < 0:
        # nothing set. process all
        bids = list(range(len(plt._s["boundary_splines"])))
        # and also create boundary linkage
        plt._s["boundary_connection"] = _BoundaryConnection(
            plt._s["boundary_splines"]
        )

    else:
        bids.append(bid)

    # process boundaries. cps as well, if needed
    boundary_splines = plt._s["boundary_splines"]
    bc_areas = plt._s["boundary_spline_areas"]
    cp_id = plt._s["picked_cp_id"]
    for b in bids:
        b_spl = boundary_splines[b]
        show_o = b_spl.show_options
        show_o["resolutions"] = plt._c["sample_resolutions"]
        show_o["control_points"] = False
        show_o["control_mesh"] = True
        show_o["lighting"] = "off"
        actors = b_spl.showable()
        # don't want this to be pickable
        for a in actors.values():
            a.pickable(False)

        bc_area = bc_areas[b]
        cp_half = int(len(bc_area.cps) / 2)
        bc_area.cps[:cp_half] = b_spl.cps
        if bc_area.dim == 2:  # noqa PLR2004
            actor = bc_area.extract.faces(
                plt._c["sample_resolutions"]
            ).showable()
        elif bc_area.dim == 3:  # noqa PLR2004
            actor = bc_area.extract.volumes(
                plt._c["sample_resolutions"]
            ).showable()
        actor.c("pink").lighting("off").alpha(0.8).pickable(False)

        # now add
        plt._s["boundary_actors"][b] = actors
        plt._s["boundary_area_actors"][b] = actor

        # process cps
        # nothing selected -> update all
        if cp_id < 0:
            new_cps = []
            for i, cp in enumerate(b_spl.cps):
                sph = vedo.Sphere(cp, **_CONF["sphere_option"])
                sph.cp_id = i
                sph.boundary_id = b
                new_cps.append(sph)
            plt._s["boundary_cp_actors"][b] = new_cps
            continue

        sphere_mesh = plt._s["boundary_cp_actors"][b][cp_id]
        sphere_mesh.pos(*b_spl.cps[cp_id])
        sphere_mesh.apply_transform_from_actor()


def _process_parametric_view(plt):
    """ """
    p_view = plt._s["parametric_view"]
    show_o = p_view.show_options
    show_o["alpha"] = 0.2
    show_o["c"] = "yellow"
    show_o["lighting"] = "off"
    actors = p_view.showable(as_dict=True)
    for a in actors.values():
        a.pickable(False)
        if isinstance(a, vedo.Assembly):
            for obj in a.recursive_unpack():
                obj.pickable(False)
    plt._s["parametric_view_actors"] = actors


def _bid_to_dof(spl, bid):
    """ """
    ortho_dim, extrema = divmod(bid, spl.para_dim)

    all_ = slice(None)
    indices = [all_] * spl.para_dim
    indices[ortho_dim] = int(extrema * -1)

    dir_flip = -1.0 if extrema else 1.0

    return spl.multi_index[tuple(indices)], ortho_dim, dir_flip


def refine(spl):
    """
    A predefined refinement rule, so that you can apply to both field and
    boundaries.
    """
    spl.elevate_degrees(list(range(spl.para_dim)) * 2)
    n_new_kvs = 10
    for i, (l_bound, u_bound) in enumerate(zip(*spl.parametric_bounds)):
        spl.insert_knots(
            i,
            np.linspace(l_bound, u_bound, int(n_new_kvs + 2))[1:-1],
        )
    return spl


class Poisson2D(vedo.Plotter, FeigenBase):
    """
    Self contained interactive plotter base.
    Can move control points on the left plot and the right plot will show
    the solution of simple laplace problem

    Parameters
    ----------
    uri: str
    degree: int
    ncoeffs: list
      list of int
    """

    __slots__ = ("_c", "_s")

    def __init__(self, spline=None):  # noqa PLR0915
        """
        Create spline and setup callbacks
        """
        # dict to hold all the configs
        # will hold initial configuration constants
        # this won't be actively updated if anything changes
        self._c = {}

        # dict to hold all current state variables
        # this will be actively used
        self._s = {}

        # plotter initialization constants
        self._c["dim"] = 2  # 2D
        self._c[
            "n_subplots"
        ] = 3  # geometry, boundary condition, server response
        self._c["geometry_plot"] = 0
        self._c["bc_plot"] = 1
        self._c["server_plot"] = 2

        # field dim is temporary for now - (1)
        self._c["field_dim"] = 1

        # set title name
        self._c["window_title"] = "Poisson 2D"

        # sampling resolutions
        default_sampling_resolution = 50
        self._c["sample_resolutions"] = default_sampling_resolution

        # now init
        super().__init__(
            N=self._c["n_subplots"],
            interactive=False,  # True will pop the window already
            sharecam=False,
            title=self._c["window_title"],
        )

        # add callbacks
        self.add_callback("Interaction", self._update)
        self.add_callback("LeftButtonPress", self._left_click)
        self.add_callback("LeftButtonRelease", self._left_release)
        self.add_callback("RightButtonPress", self._right_click)

        # register spline
        if spline is None:
            self._s["spline"] = splinepy.helpme.create.surface_circle(1).nurbs
            self._s["spline"].insert_knots(0, [0.5])
            self._s["spline"].insert_knots(1, [0.5])
        else:
            support_dim = 2
            if spline.para_dim != support_dim or spline.dim != support_dim:
                raise ValueError(
                    "This plotter is built for splines with "
                    "para_dim=2 and dim=2"
                )
            self._s["spline"] = spline

        # create boundary splines - will be used to control BCs
        conform_param_view = self._s["spline"].create.parametric_view(
            axes=False, conform=True
        )
        para_boundaries = conform_param_view.extract.boundaries()
        para_dim = conform_param_view.para_dim
        dim = conform_param_view.dim
        cp_bounds = conform_param_view.control_point_bounds
        # apply small offset - 5 % of the size
        offsets = (cp_bounds[1] - cp_bounds[0]) * 0.05
        for i in range(para_dim):
            offset = np.zeros(dim)
            offset[i] = offsets[i]

            pb_low = para_boundaries[i * para_dim]
            pb_low.cps[:] -= offset

            pb_high = para_boundaries[i * para_dim + 1]
            pb_high.cps[:] += offset

        self._s["boundary_splines"] = para_boundaries
        self._s["boundary_spline_areas"] = [
            pb.create.extruded([0] * pb.dim) for pb in para_boundaries
        ]
        self._s["boundary_actors"] = [None] * len(para_boundaries)
        self._s["boundary_cp_actors"] = [None] * len(para_boundaries)
        self._s["boundary_area_actors"] = [None] * len(para_boundaries)

        # create parametric_view - this does not consider embedded geometry
        self._s["parametric_view"] = self._s["spline"].create.parametric_view(
            axes=True, conform=False
        )

        # plotter mode for 2d, trackball actor
        self._c["plotter_mode"] = "TrackballActor"

        # initialize value
        self._s["picked_cp_id"] = -1
        self._s["picked_boundary_id"] = -1

        self._logd("Finished setup.", "cs:", self._c, "s:", self._s)

    def _left_click(self, evt):
        """
        Callback for left click.
        selects and saves object id

        Parameters
        ----------
        evt: vedo.Event

        Returns
        -------
        None
        """
        # nothing selected. exit
        if not evt.actor:
            return None

        # we've assigned ids to control point spheres
        # check if this actor has this attr
        cp_id = getattr(evt.actor, "cp_id", None)

        # no cp_id means irrelevant actor
        if cp_id is None:
            return None

        # well selected
        self._s["picked_cp_id"] = cp_id
        # maybe this is boundary spline
        self._s["picked_boundary_id"] = getattr(evt.actor, "boundary_id", -1)
        # save picked at
        self._s["picked_at"] = evt.at

        # once clicked, this means server's previous response is useless
        server_actors = self._s.get("server_plot_actors", None)
        if server_actors is not None:
            self.at(self._c["server_plot"]).remove(*server_actors.values())

    def _left_release(self, evt):  # noqa ARG002
        """
        Left release unmarks picked object.

        Parameters
        ----------
        evt: vedo.Event
          unused

        Returns
        -------
        None
        """
        self._s["picked_cp_id"] = -1
        self._s["picked_boundary_id"] = -1

    def _update(self, evt):
        """
        Interaction / Drag event.
        Cleans and add updated splines.

        Parameters
        ----------
        evt: vedo.Event

        Returns
        -------
        None
        """
        # exit if there's no selection
        cp_id = self._s["picked_cp_id"]
        if cp_id < 0:
            return None

        # exit if this plot is server plot
        if evt.at == self._c["server_plot"]:
            return None

        # geometry update
        if (
            evt.at == self._c["geometry_plot"]
            and self._s["picked_at"] == evt.at
        ):
            # remove existing actors
            self.remove(*self._s["spline_actors"].values(), at=evt.at)

            # update cp
            # compute physical coordinate of the mouse
            coord = self.compute_world_coordinate(evt.picked2d, at=evt.at)[
                : self._s["spline"].dim
            ]
            self._s["spline"].cps[cp_id] = coord

            # prepare actors
            _process_spline_actors(self)

            # add updated splines
            self.add(*self._s["spline_actors"].values(), at=evt.at)

        # boundary condition update
        elif evt.at == self._c["bc_plot"] and self._s["picked_at"] == evt.at:
            bid = self._s["picked_boundary_id"]
            boundary_spline = self._s["boundary_splines"][bid]

            # first remove
            self.remove(*self._s["boundary_actors"][bid].values(), at=evt.at)
            self.remove(self._s["boundary_area_actors"][bid], at=evt.at)

            # compute physical coordinate of the mouse
            # depends on the field dim, we will have to restrict
            # movement
            if self._c["field_dim"] == 1:  # noqa PLR2004
                ind = int(bid / self._s["spline"].para_dim)
            elif self._c["field_dim"] == 2:  # noqa PLR2004
                ind = slice(None, None, None)
            else:
                raise ValueError("This interactor supports field_dim < 3.")

            coord = self.compute_world_coordinate(evt.picked2d, at=evt.at)[
                : boundary_spline.dim
            ][ind]
            self._s["boundary_splines"][bid].cps[cp_id, ind] = coord

            # process and add
            _process_boundary_actors(self)

            # check connection
            connec_bid, connec_cp_id = self._s[
                "boundary_connection"
            ].transform_ids(bid, cp_id)

            # there's connection - update connected boundary as well
            if connec_bid is not None:
                coupled_cp_position = self._s[
                    "boundary_connection"
                ].transform_position(
                    connec_bid,
                    connec_cp_id,
                    bid,
                    cp_id,
                    self._s["boundary_splines"][bid].cps[cp_id],
                )

                # update coupled cp
                self._s["boundary_splines"][connec_bid].cps[
                    connec_cp_id
                ] = coupled_cp_position

                # temporarily overwrite picked_id for coupled update
                self._s["picked_cp_id"] = connec_cp_id
                self._s["picked_boundary_id"] = connec_bid
                # update
                self.remove(
                    *self._s["boundary_actors"][connec_bid].values(),
                    at=evt.at,
                )
                self.remove(
                    self._s["boundary_area_actors"][connec_bid], at=evt.at
                )
                _process_boundary_actors(self)
                self.add(
                    *self._s["boundary_actors"][connec_bid].values(),
                    at=evt.at,
                )
                self.add(
                    self._s["boundary_area_actors"][connec_bid], at=evt.at
                )
                # reset
                self._s["picked_cp_id"] = cp_id
                self._s["picked_boundary_id"] = bid

            self.add(*self._s["boundary_actors"][bid].values(), at=evt.at)
            self.add(self._s["boundary_area_actors"][bid], at=evt.at)

        # render!
        self.render()

    def _right_click(self, evt):  # noqa ARG002
        """
        Syncs solution on right click.

        Parameters
        ----------
        evt: vedo.Event

        Returns
        -------
        None
        """
        # remove
        server_actors = self._s.get("server_plot_actors", None)
        if server_actors is not None:
            self.remove(*server_actors.values(), at=self._c["server_plot"])

        # we will solve the problem with collocation methods
        geometry = self._s["spline"]
        solution = self._s.get("solution_spline", None)
        if solution is None:
            dict_spline = geometry.current_core_properties()
            dict_spline["control_points"] = np.zeros(
                (len(geometry.cps), 1), dtype="float64"
            )
            solution = type(geometry)(**dict_spline)
            self._s["solution_spline"] = solution

            refine(solution)

            self._logd(
                f"created solution spline - degrees: {solution.ds}"
                f"knot_vectors: {solution.kvs}"
            )

            # get greville points and sparsity pattern
            queries = solution.greville_abscissae()
            self._s["solution_queries"] = queries

            support = solution.support(queries)
            n_row, n_col = support.shape
            row_ids = np.arange(n_row).repeat(n_col)
            col_ids = support.ravel()
            self._s["solution_n_row"] = n_row
            self._s["solution_row_ids"] = row_ids
            self._s["solution_n_col"] = n_col
            self._s["solution_col_ids"] = col_ids

        queries = self._s["solution_queries"]

        # get mapper and evaluate basis laplacian
        mapper = solution.mapper(reference=geometry)
        b_laplacian, _ = mapper.basis_laplacian_and_support(queries)

        # prepare lhs sparse matrix
        sp_laplacian = csr_array(
            (
                b_laplacian.ravel(),
                (
                    self._s["solution_row_ids"],
                    self._s["solution_col_ids"],
                ),
            ),
            shape=[self._s["solution_n_row"]] * 2,
        )

        # prepare rhs
        # we could add source function here
        mi = solution.multi_index
        boundary_dofs = np.concatenate((mi[[0, -1], :], mi[1:-1, [0, -1]]))
        rhs = np.ones(len(queries))
        rhs[boundary_dofs] = 0.0

        # set zeros and ones on lhs
        sp_laplacian[boundary_dofs] *= 0
        sp_laplacian[boundary_dofs, boundary_dofs] = 1

        # apply boundary condition
        # we will loop and extract boundary values
        for bid, (bdr_spline, bdr_area) in enumerate(
            zip(
                self._s["boundary_splines"],
                self._s["boundary_spline_areas"],
            )
        ):
            # now, based on bdr_area, we extract values for boundary condition
            cp_half = int(len(bdr_area.cps) / 2)

            bdr = bdr_spline.copy()
            bdr.cps[:] = bdr_area.cps[:cp_half] - bdr_area.cps[cp_half:]
            # self._s["solution_refinement"](bdr)
            refine(bdr)
            b_dof, axis, factor = _bid_to_dof(solution, bid)
            rhs[b_dof] = bdr.cps[:, axis] * factor

        solution.cps[:] = linalg.spsolve(-sp_laplacian, rhs).reshape(-1, 1)

        # eval field - currently, th
        field = solution.sample(self._c["sample_resolutions"])

        # update vedo object
        # we should copy the geometry (spline),
        # because we don't want two plots to have same color
        geo_actor = self._s["spline_actors"]["spline"].clone()
        geo_actor.cmap("plasma", field).alpha(0.9).add_scalarbar3d()
        self._s["server_plot_actors"] = {
            "spline": geo_actor,
            "knots": self._s["spline_actors"]["knots"],
        }

        # for some reason add() won't work here
        self.show(
            *self._s["server_plot_actors"].values(),
            at=self._c["server_plot"],
            mode=self._c["plotter_mode"],
        )

    def start(self):
        """
        Starts interactive move

        Returns
        -------
        plot: BSpline2D
          self
        """
        # process for the first time
        _process_spline_actors(self)
        _process_boundary_actors(self)
        _process_parametric_view(self)

        # show everything
        self.show(
            "Geometry",
            *self._s["spline_actors"].values(),
            *self._s["spline_cp_actors"],
            at=self._c["geometry_plot"],
            interactive=False,
            mode=self._c["plotter_mode"],
        )
        for b, b_cp in zip(
            self._s["boundary_actors"], self._s["boundary_cp_actors"]
        ):
            self.show(
                *b.values(),
                *b_cp,
                at=self._c["bc_plot"],
                interactive=False,
                mode=self._c["plotter_mode"],
            )

        self.show(
            "Boundary condition in parametric view",
            *self._s["parametric_view_actors"].values(),
            at=self._c["bc_plot"],
            interactive=False,
            mode=self._c["plotter_mode"],
        )

        self.show(
            "Solution - right click to sync",
            at=self._c["server_plot"],
            interactive=False,
            mode=self._c["plotter_mode"],
        )

        # let's start
        self.interactive()

        return self
