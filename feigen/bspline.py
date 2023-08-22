import numpy as np
import splinepy
import vedo

from feigen import comm
from feigen._base import FeigenBase

# skip bound checks
splinepy.settings.CHECK_BOUNDS = False


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
            sph = vedo.Sphere(cp, r=0.05, res=5)
            sph.cp_id = i
            new_cps.append(sph)
        plt._state["spline_cp_actors"] = new_cps
        return None

    # now for specific cp
    sph = vedo.Sphere(plt._state["spline"].cps[cp_id], r=0.05, res=5)
    sph.cp_id = cp_id
    plt._state["spline_cp_actors"][cp_id] = sph


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

    else:
        bids.append(bid)

    # process boundaries. cps as well, if needed
    boundary_splines = plt._state["boundary_splines"]
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

        # now add
        plt._state["boundary_actors"][b] = actors

        # process cps
        # nothing selected -> update all
        if cp_id < 0:
            new_cps = []
            for i, cp in enumerate(b_spl.cps):
                sph = vedo.Sphere(cp, r=0.05, res=5)
                sph.cp_id = i
                sph.boundary_id = b
                new_cps.append(sph)
            plt._state["boundary_cp_actors"][b] = new_cps
            continue

        # now for specific cp
        sph = vedo.Sphere(b_spl.cps[cp_id], r=0.05, res=5)
        sph.cp_id = cp_id
        sph.boundary_id = b
        plt._state["boundary_cp_actors"][b][cp_id] = sph


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

    def __init__(self, uri, degree=None, ncoeffs=None):
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
        self._config["dim"] = int(2)  # 2D
        self._config["n_subplots"] = int(
            3
        )  # geometry, boundary condition, server response
        self._config["geometry_plot"] = int(0)
        self._config["bc_plot"] = int(1)
        self._config["server_plot"] = int(2)

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

        # add a sync button
        # self.at(self._config["server_plot"]).add_button(
        #    self._iganet_sync,
        #    pos=(0.7, 0.05),  # x, y fraction from bottom left corner
        #    states=["sync"],  # only one state
        #    c=["w"],  # TODO - probably just need one
        #    bc=["dg"],  # TODO  - probably just need one
        #    font="courier",  # arial, courier, times
        #    size=25,
        #    bold=True,
        #    italic=False,
        # )

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
        self._state["boundary_actors"] = [None] * len(para_boundaries)
        self._state["boundary_cp_actors"] = [None] * len(para_boundaries)

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
        if evt.at == self._config["geometry_plot"]:
            # remove existing actors
            self.remove(*self._state["spline_actors"].values(), at=evt.at)
            self.remove(
                self._state["spline_cp_actors"][cp_id],
                at=evt.at,
            )

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
            self.add(self._state["spline_cp_actors"][cp_id], at=evt.at)

        # boundary condition update
        elif evt.at == self._config["bc_plot"]:
            bid = self._state["picked_boundary_id"]
            boundary_spline = self._state["boundary_splines"][bid]

            # first remove
            self.remove(
                *self._state["boundary_actors"][bid].values(), at=evt.at
            )
            self.remove(
                self._state["boundary_cp_actors"][bid][cp_id], at=evt.at
            )

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

            self.add(*self._state["boundary_actors"][bid].values(), at=evt.at)
            self.add(self._state["boundary_cp_actors"][bid][cp_id], at=evt.at)

        # render!
        self.render()

    def _iganet_sync(self):
        pass

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

        # show everything
        self.show(
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

        # let's start
        self.show(interactive=True)

        # TODO - close websocket here?
        self._config["iganet_ws"].websocket.close()

        return self
