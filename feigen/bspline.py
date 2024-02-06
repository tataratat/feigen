import numpy as np
import splinepy
import vedo
from gustaf.utils.arr import enforce_len

from feigen import comm
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
    spl = plt._state["spline"]
    show_options = spl.show_options
    show_options["resolutions"] = plt._config["sample_resolutions"]
    show_options["control_points"] = False
    show_options["control_mesh"] = True
    show_options["lighting"] = "off"

    # get spline actors
    plt._state["spline_actors"] = spl.showable()

    # don't want any of these actors to be pickable
    for v in plt._state["spline_actors"].values():
        v.pickable(False)

    # process cps
    # nothing selected -> update all
    cp_id = plt._state["picked_cp_id"]
    new_cps = []
    if cp_id < 0:
        for i, cp in enumerate(plt._state["spline"].cps):
            sph = vedo.Sphere(cp, **_CONF["sphere_option"])
            sph.cp_id = i
            new_cps.append(sph)
        plt._state["spline_cp_actors"] = new_cps
        return None

    sphere_mesh = plt._s["spline_cp_actors"][cp_id]
    sphere_mesh.pos(*plt._s["spline"].cps[cp_id])
    sphere_mesh.apply_transform_from_actor()


def _process_boundary_actors(plt):
    """
    Helper function to process boundary spline.
    all necessary values should be available from _state

    Parameters
    ----------
    plt: BSpline2D

    Returns
    -------
    None
    """
    # get boundary_spline
    bid = plt._state["picked_boundary_id"]

    bids = []

    if bid < 0:
        # nothing set. process all
        bids = list(range(len(plt._state["boundary_splines"])))
        # and also create boundary linkage
        plt._state["boundary_connection"] = _BoundaryConnection(
            plt._state["boundary_splines"]
        )

    else:
        bids.append(bid)

    # process boundaries. cps as well, if needed
    boundary_splines = plt._state["boundary_splines"]
    bc_areas = plt._state["boundary_spline_areas"]
    cp_id = plt._state["picked_cp_id"]
    for b in bids:
        b_spl = boundary_splines[b]
        show_o = b_spl.show_options
        show_o["resolutions"] = plt._config["sample_resolutions"]
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
                plt._config["sample_resolutions"]
            ).showable()
        elif bc_area.dim == 3:  # noqa PLR2004
            actor = bc_area.extract.volumes(
                plt._config["sample_resolutions"]
            ).showable()
        actor.c("pink").lighting("off").alpha(0.8).pickable(False)

        # now add
        plt._state["boundary_actors"][b] = actors
        plt._state["boundary_area_actors"][b] = actor

        # process cps
        # nothing selected -> update all
        if cp_id < 0:
            new_cps = []
            for i, cp in enumerate(b_spl.cps):
                sph = vedo.Sphere(cp, **_CONF["sphere_option"])
                sph.cp_id = i
                sph.boundary_id = b
                new_cps.append(sph)
            plt._state["boundary_cp_actors"][b] = new_cps
            continue

        sphere_mesh = plt._s["boundary_cp_actors"][b][cp_id]
        sphere_mesh.pos(*b_spl.cps[cp_id])
        sphere_mesh.apply_transform_from_actor()


def _process_parametric_view(plt):
    """ """
    p_view = plt._state["parametric_view"]
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
    plt._state["parametric_view_actors"] = actors


class BSpline2D(vedo.Plotter, FeigenBase):
    """
    Self contained interactive plotter base.
    Can move control points on the left plot and the right plot will show
    the response from iganet.

    Upon initialization, it will create bspline at iganet server and
    setup interactive plotter.

    Parameters
    ----------
    uri: str
    degree: int
    ncoeffs: list
      list of int
    """

    __slots__ = ("_config", "_state")

    def __init__(self, uri, degree=None, ncoeffs=None):  # noqa PLR0915
        """
        Create spline and setup callbacks
        """
        # dict to hold all the configs
        # will hold initial configuration constants
        # this won't be actively updated if anything changes
        self._config = {}

        # dict to hold all current state variables
        # this will be actively used
        self._state = {}

        # plotter initialization constants
        self._config["dim"] = 2  # 2D
        self._config[
            "n_subplots"
        ] = 3  # geometry, boundary condition, server response
        self._config["geometry_plot"] = 0
        self._config["bc_plot"] = 1
        self._config["server_plot"] = 2

        # field dim is temporary for now - (1)
        self._config["field_dim"] = 1

        # prepare degrees
        if degree is None:
            degree = 2
            self._logd(f"`degree` not specified, setting default ({degree})")

        # make sure degree is int
        degree = int(degree)

        # prepare ncoeffs
        if ncoeffs is None:
            ncoeffs = [degree + 2] * self._config["dim"]
            self._logd(
                "`ncoeffs` not specified, setting default (degree + 2) "
                f"(= {ncoeffs})"
            )

        # make sure ncoeff is list
        ncoeffs = list(ncoeffs)

        # set title name
        self._config["window_title"] = "IgaNet BSpline 2D"

        # sampling resolutions
        default_sampling_resolution = 50
        self._config["sample_resolutions"] = default_sampling_resolution

        # now init
        super().__init__(
            N=self._config["n_subplots"],
            interactive=False,  # True will pop the window already
            sharecam=False,
            title=self._config["window_title"],
        )

        # add callbacks
        self.add_callback("Interaction", self._update)
        self.add_callback("LeftButtonPress", self._left_click)
        self.add_callback("LeftButtonRelease", self._left_release)
        self.add_callback("RightButtonPress", self._iganet_sync)

        # now, connect to websockets
        self._config["iganet_ws"] = comm.WebSocketClient(uri)

        # create session - returns session id
        self._config["session_id"] = comm.RequestForm.create_session(
            self._config["iganet_ws"]
        )

        # with session id, create request form
        self._config["form"] = comm.RequestForm(
            self._config["iganet_ws"], self._config["session_id"]
        )

        # create spline - returns spline id
        self._config["spline_id"] = self._config["form"].create_spline(
            "BSplineSurface", degree=degree, ncoeffs=ncoeffs
        )

        # create matching local spline
        self._state["spline"] = splinepy.BSpline(
            **self._config["form"].model_info(
                self._config["spline_id"], to_splinepy_dict=True
            )
        )

        # create boundary splines - will be used to control BCs
        conform_param_view = self._state["spline"].create.parametric_view(
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

        self._state["boundary_splines"] = para_boundaries
        self._state["boundary_spline_areas"] = [
            pb.create.extruded([0] * pb.dim) for pb in para_boundaries
        ]
        self._state["boundary_actors"] = [None] * len(para_boundaries)
        self._state["boundary_cp_actors"] = [None] * len(para_boundaries)
        self._state["boundary_area_actors"] = [None] * len(para_boundaries)

        # create parametric_view - this does not consider embedded geometry
        self._state["parametric_view"] = self._state[
            "spline"
        ].create.parametric_view(axes=True, conform=False)

        # plotter mode for 2d, trackball actor
        self._config["plotter_mode"] = "TrackballActor"

        # initialize value
        self._state["picked_cp_id"] = -1
        self._state["picked_boundary_id"] = -1

        self._logd(
            "Finished setup.", "configs:", self._config, "state:", self._state
        )

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
        self._state["picked_cp_id"] = cp_id
        # maybe this is boundary spline
        self._state["picked_boundary_id"] = getattr(
            evt.actor, "boundary_id", -1
        )
        # save picked at
        self._state["picked_at"] = evt.at

        # once clicked, this means server's previous response is useless
        server_actors = self._state.get("server_plot_actors", None)
        if server_actors is not None:
            self.at(self._config["server_plot"]).remove(
                *server_actors.values()
            )

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
        self._state["picked_cp_id"] = -1
        self._state["picked_boundary_id"] = -1

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
        cp_id = self._state["picked_cp_id"]
        if cp_id < 0:
            return None

        # exit if this plot is server plot
        if evt.at == self._config["server_plot"]:
            return None

        # geometry update
        if (
            evt.at == self._config["geometry_plot"]
            and self._state["picked_at"] == evt.at
        ):
            # remove existing actors
            self.remove(*self._state["spline_actors"].values(), at=evt.at)

            # update cp
            # compute physical coordinate of the mouse
            coord = self.compute_world_coordinate(evt.picked2d, at=evt.at)[
                : self._state["spline"].dim
            ]
            self._state["spline"].cps[cp_id] = coord

            # prepare actors
            _process_spline_actors(self)

            # add updated splines
            self.add(*self._state["spline_actors"].values(), at=evt.at)

        # boundary condition update
        elif (
            evt.at == self._config["bc_plot"]
            and self._state["picked_at"] == evt.at
        ):
            bid = self._state["picked_boundary_id"]
            boundary_spline = self._state["boundary_splines"][bid]

            # first remove
            self.remove(
                *self._state["boundary_actors"][bid].values(), at=evt.at
            )
            self.remove(self._state["boundary_area_actors"][bid], at=evt.at)

            # compute physical coordinate of the mouse
            # depends on the field dim, we will have to restrict
            # movement
            if self._config["field_dim"] == 1:  # noqa PLR2004
                ind = int(bid / self._state["spline"].para_dim)
            elif self._config["field_dim"] == 2:  # noqa PLR2004
                ind = slice(None, None, None)
            else:
                raise ValueError("This interactor supports field_dim < 3.")

            coord = self.compute_world_coordinate(evt.picked2d, at=evt.at)[
                : boundary_spline.dim
            ][ind]
            self._state["boundary_splines"][bid].cps[cp_id, ind] = coord

            # process and add
            _process_boundary_actors(self)

            # check connection
            connec_bid, connec_cp_id = self._state[
                "boundary_connection"
            ].transform_ids(bid, cp_id)

            # there's connection - update connected boundary as well
            if connec_bid is not None:
                coupled_cp_position = self._state[
                    "boundary_connection"
                ].transform_position(
                    connec_bid,
                    connec_cp_id,
                    bid,
                    cp_id,
                    self._state["boundary_splines"][bid].cps[cp_id],
                )

                # update coupled cp
                self._state["boundary_splines"][connec_bid].cps[
                    connec_cp_id
                ] = coupled_cp_position

                # temporarily overwrite picked_id for coupled update
                self._state["picked_cp_id"] = connec_cp_id
                self._state["picked_boundary_id"] = connec_bid
                # update
                self.remove(
                    *self._state["boundary_actors"][connec_bid].values(),
                    at=evt.at,
                )
                self.remove(
                    self._state["boundary_area_actors"][connec_bid], at=evt.at
                )
                _process_boundary_actors(self)
                self.add(
                    *self._state["boundary_actors"][connec_bid].values(),
                    at=evt.at,
                )
                self.add(
                    self._state["boundary_area_actors"][connec_bid], at=evt.at
                )
                # reset
                self._state["picked_cp_id"] = cp_id
                self._state["picked_boundary_id"] = bid

            self.add(*self._state["boundary_actors"][bid].values(), at=evt.at)
            self.add(self._state["boundary_area_actors"][bid], at=evt.at)

        # render!
        self.render()

    def _iganet_sync(self, evt):  # noqa ARG002
        """
        Syncs iganet on right click.

        Parameters
        ----------
        evt: vedo.Event

        Returns
        -------
        None
        """
        # remove
        server_actors = self._state.get("server_plot_actors", None)
        if server_actors is not None:
            self.remove(
                *server_actors.values(), at=self._config["server_plot"]
            )

        # sync current coordinates
        self._config["form"].sync_coeffs(
            self._config["spline_id"], self._state["spline"].cps
        )

        # sync boundary condition
        # here

        # eval field - currently, th
        field = self._config["form"].evaluate_model(
            self._config["spline_id"],
            "ValueFieldMagnitude",
            enforce_len(
                self._config["sample_resolutions"],
                self._state["spline"].para_dim,
            ).tolist(),
        )  # this is ravel

        # update vedo object
        # we should copy the geometry (spline),
        # because we don't want two plots to have same color
        geo_actor = self._state["spline_actors"]["spline"].clone()
        geo_actor.cmap("plasma", field).alpha(0.9)
        self._state["server_plot_actors"] = {
            "spline": geo_actor,
            "knots": self._state["spline_actors"]["knots"],
        }
        self.add(
            *self._state["server_plot_actors"].values(),
            at=self._config["server_plot"],
        )

        # eval current geometry
        geo_eval = self._config["form"].evaluate_model(
            self._config["spline_id"],
            "ValueField",
            enforce_len(
                int(self._config["sample_resolutions"] / 5),
                self._state["spline"].para_dim,
            ).tolist(),
        )
        eval_points = vedo.Points(geo_eval, c="white")
        eval_point_ids = eval_points.labels("id", on="points", font="VTK")
        self._state["server_plot_actors"]["evaluated_points"] = eval_points
        self._state["server_plot_actors"][
            "evaluated_point_ids"
        ] = eval_point_ids.c("grey")

        # for some reason add won't work here
        self.show(
            *self._state["server_plot_actors"].values(),
            at=self._config["server_plot"],
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
            *self._state["spline_actors"].values(),
            *self._state["spline_cp_actors"],
            at=self._config["geometry_plot"],
            interactive=False,
            mode=self._config["plotter_mode"],
        )
        for b, b_cp in zip(
            self._state["boundary_actors"], self._state["boundary_cp_actors"]
        ):
            self.show(
                *b.values(),
                *b_cp,
                at=self._config["bc_plot"],
                interactive=False,
                mode=self._config["plotter_mode"],
            )

        self.show(
            "Boundary condition in parametric view",
            *self._state["parametric_view_actors"].values(),
            at=self._config["bc_plot"],
            interactive=False,
            mode=self._config["plotter_mode"],
        )

        self.show(
            "IgaNet server - right click to sync",
            at=self._config["server_plot"],
            interactive=False,
            mode=self._config["plotter_mode"],
        )

        # let's start
        self.interactive()

        # TODO - close websocket here?
        self._config["iganet_ws"].websocket.close()

        return self
