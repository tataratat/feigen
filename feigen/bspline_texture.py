import splinepy
import vedo

from feigen._base import FeigenBase

# skip bound checks
splinepy.settings.CHECK_BOUNDS = False
splinepy.settings.NTHREADS = 2

_CONF = {
    "sphere_option": {
        "res": 8,
        "r": 0.02,
        "c": "red",
    }
}


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
    spline_actors = plt._s.get("spline_actors", None)
    plt._s["spline_actors"] = spl.showable()
    if spline_actors is None:
        plt._s["spline_actors"]["spline"].texture(plt._c["texture_uri"])
    else:
        spline_actors["spline"].vertices = plt._s["spline_actors"][
            "spline"
        ].vertices
        plt._s["spline_actors"]["spline"] = spline_actors["spline"]

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


class BSplineTexture(vedo.Plotter, FeigenBase):
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

    def __init__(
        self,
        spline=None,
        texture_uri=None,
    ):
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
        self._c["n_subplots"] = 1  # geometry
        self._c["geometry_plot"] = 0

        # field dim is temporary for now - (1)
        self._c["field_dim"] = 1

        # set title name
        self._c["window_title"] = "Smile"

        # sampling resolutions
        default_sampling_resolution = 50
        self._c["sample_resolutions"] = default_sampling_resolution

        # save texture path
        self._c["texture_uri"] = texture_uri

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
            self._s["spline"] = splinepy.helpme.create.box(1, 1).bspline
            self._s["spline"].insert_knots(0, [0.37, 0.62])
            self._s["spline"].insert_knots(1, [0.35, 0.5])
            self._s["spline"].elevate_degrees([0])
            self._s["spline"].show_options["knots"] = False
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

    def _right_click(self, evt):  # noqa: ARG002
        """
        Ends all and just show what we have
        """
        if self.state % 2 == 0:
            self.add(*self._s["spline_cp_actors"])
            tmp = self._s["spline_actors"].copy()
            tmp.pop("spline")
            self.add(*tmp.values())
        else:
            self.clear()
            self.add(self._s["spline_actors"]["spline"])

        self.state += 1
        self.render()

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
            "Right click to hide/show control mesh",
            self._s["spline_actors"]["spline"],
            at=self._c["geometry_plot"],
            interactive=False,
            mode=self._c["plotter_mode"],
            bg="grey",
        )

        self.state = 0

        # let's start
        self.interactive()

        return self
