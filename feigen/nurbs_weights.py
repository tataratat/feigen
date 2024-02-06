import numpy as np
import splinepy
import vedo

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


_W_MAX = 5
_W_MIN = -1.5


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
            color = vedo.color_map(
                plt._s["spline"].ws[i], "rainbow", _W_MIN, _W_MAX
            )
            sph.c(*color)

        plt._s["spline_cp_actors"] = new_cps
        return None

    sphere_mesh = plt._s["spline_cp_actors"][cp_id]
    sphere_mesh.pos(*plt._s["spline"].cps[cp_id])
    sphere_mesh.apply_transform_from_actor()
    # set color
    color = vedo.color_map(
        plt._s["spline"].ws[cp_id], "rainbow", _W_MIN, _W_MAX
    )
    sphere_mesh.c(*color)


class NURBSWeights(vedo.Plotter, FeigenBase):
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
        ] = 2  # geometry, boundary condition, server response
        self._c["geometry_plot"] = 0
        self._c["server_plot"] = 1

        # field dim is temporary for now - (1)
        self._c["field_dim"] = 1

        # set title name
        self._c["window_title"] = "NURBS Weights"

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
        else:
            support_dim = 2
            if spline.para_dim != support_dim or spline.dim != support_dim:
                raise ValueError(
                    "This plotter is built for splines with "
                    "para_dim=2 and dim=2"
                )
            self._s["spline"] = spline

        # plotter mode for 2d, trackball actor
        self._c["plotter_mode"] = "TrackballActor"

        # initialize value
        self._s["picked_cp_id"] = -1
        self._s["picked_boundary_id"] = -1

        self._logd("Finished setup.", "cs:", self._c, "s:", self._s)

    def _weight_slider(self, widget, evt):
        """ """
        weight = widget.value
        if np.isclose(weight, 0.0):
            return None
        latest_cp = self._s.get("latest_cp_id", None)
        if latest_cp is not None:
            self._s["spline"].ws[latest_cp] = weight
            self._s["picked_cp_id"] = latest_cp
            self._update(evt)
            self._s["picked_cp_id"] = -1

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
        self._s["latest_cp_id"] = cp_id  # we keep this for the slider
        # save picked at
        self._s["picked_at"] = evt.at

        # we update widget value accordingly
        self.sliders[0][0].value = self._s["spline"].ws[cp_id]

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
        if getattr(evt, "at", None) == self._c["server_plot"]:
            return None

        # geometry update
        if isinstance(evt, str) or (
            evt.at == self._c["geometry_plot"]
            and self._s["picked_at"] == evt.at
        ):
            # remove existing actors
            self.remove(
                *self._s["spline_actors"].values(), at=self._c["geometry_plot"]
            )

            # update cp
            # compute physical coordinate of the mouse
            if not isinstance(evt, str):
                coord = self.compute_world_coordinate(evt.picked2d, at=evt.at)[
                    : self._s["spline"].dim
                ]
                self._s["spline"].cps[cp_id] = coord

            # prepare actors
            _process_spline_actors(self)

            # add updated splines
            self.add(
                *self._s["spline_actors"].values(), at=self._c["geometry_plot"]
            )

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

        # eval field - currently, th
        field = geometry.create.determinant_spline().sample(
            self._c["sample_resolutions"]
        )

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

        # show everything

        self.show(
            "Geometry",
            *self._s["spline_actors"].values(),
            *self._s["spline_cp_actors"],
            at=self._c["geometry_plot"],
            interactive=False,
            mode=self._c["plotter_mode"],
            bg="grey",
        )

        self.show(
            "Solution - right click to sync",
            at=self._c["server_plot"],
            interactive=False,
            mode=self._c["plotter_mode"],
        )

        # 2d slider somehow doesn't work, so 3D
        cp_bb = self._s["spline"].control_point_bounds
        bb_diff = cp_bb[1] - cp_bb[0]
        pos1 = cp_bb[0] - [0, bb_diff[1] / 10]
        pos2 = cp_bb[0].copy()
        pos2[0] = cp_bb[1][0]
        pos2[1] = pos1[1]
        self.at(self._c["geometry_plot"]).add_slider3d(
            self._weight_slider,
            pos1=[*pos1, 0],
            pos2=[*pos2, 0],
            xmin=_W_MIN,
            xmax=_W_MAX,
            value=1,
            s=0.04,
            title="weight",
        )

        # let's start
        self.interactive()

        return self
