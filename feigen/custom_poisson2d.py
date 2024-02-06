import numpy as np
import splinepy
import vedo
from scipy.sparse import csr_array, linalg

from feigen.poisson2d import (
    Poisson2D,
    _bid_to_dof,
    _process_boundary_actors,
    _process_parametric_view,
    _process_spline_actors,
    refine,
)

# skip bound checks
splinepy.settings.CHECK_BOUNDS = False
splinepy.settings.NTHREADS = 2


class CustomPoisson2D(Poisson2D):
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

    def __init__(  # noqa PLR0915
        self, spline=None, collocation_function=None, refinement_function=None
    ):
        """
        Create spline and setup callbacks

        Parameters
        ----------
        spline: BSpline or NURBS
          Default is a disk
        collocation_function: callable
          Function to create collocation point.
          f(spline) -> array-like collocation_points
        refinement_function: callable
          Function to refine solution spline.
          f(spline) -> None
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
        # geometry, boundary condition, server response
        self._c["n_subplots"] = 4
        self._c["geometry_plot"] = 0
        self._c["bc_plot"] = 1
        self._c["server_plot"] = 2
        self._c["error_plot"] = 3

        # field dim is temporary for now - (1)
        self._c["field_dim"] = 1

        # set title name
        self._c["window_title"] = "Custom Poisson 2D"

        # sampling resolutions
        default_sampling_resolution = 50
        self._c["sample_resolutions"] = default_sampling_resolution

        # now use vedo Plotter's init
        vedo.Plotter.__init__(
            self,
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

        # configure options
        # default spline
        # normalize!
        if spline is None:
            self._s["spline"] = splinepy.helpme.create.disk(2, 1, 90, 1).nurbs
            self._s["spline"].insert_knots(0, [0.5])
            self._s["spline"].insert_knots(1, [0.5])
            self._s["spline"].normalize_knot_vectors()
        else:  # check dims of spline
            support_dim = 2
            if spline.para_dim != support_dim or spline.dim != support_dim:
                raise ValueError(
                    "This plotter is built for splines with "
                    "para_dim=2 and dim=2"
                )
            if not isinstance(spline, splinepy.bspline.BSplineBase):
                raise TypeError("This app only supports BSpline family")
            self._s["spline"] = spline

        # collocation point generation function
        # default is greville_abscissae
        if collocation_function is None:
            self._c["collocate"] = type(self._s["spline"]).greville_abscissae
        else:
            # minimum check of "callability:
            if not callable(collocation_function):
                raise TypeError("collocation_function should be a callable")
            self._c["collocate"] = collocation_function

        # refinement function
        if refinement_function is None:
            # default refinement function raises degree twice and
            # inserts 8 uniformly distributed knots
            self._c["refine"] = refine
        else:
            # minimum check of "callability:
            if not callable(refinement_function):
                raise TypeError("refinement_function should be a callable")
            self._c["refine"] = refinement_function

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
        self._s["right_click_counter"] = 0

        self._logd("Finished setup.", "cs:", self._c, "s:", self._s)

    def _left_click(self, evt):  # noqa ARG002
        """
        Callback for left click.
        Same as Poisson2D, besides the part where we reset right click counter
        """
        self._s["right_click_counter"] = 0
        super()._left_click(evt)

        # clear error plot too
        error_actors = self._s.get("error_plot_actors", None)
        if error_actors is not None:
            self.remove(*error_actors.values(), at=self._c["error_plot"])

    def _right_click(self, evt):  # noqa ARG002
        """
        Syncs solution on right click.
        On second click,

        Parameters
        ----------
        evt: vedo.Event

        Returns
        -------
        None
        """
        # check right click counter
        # if it is 1, it means there's computation and we want to plot
        # collocation points
        if self._s["right_click_counter"] == 1:
            sol = self._s["solution_spline"]
            geo = self._s["spline"]
            # collocation points in physical space
            collocation_points = geo.evaluate(self._c["collocate"](sol))
            # if points are not so visible, we can go for spheres
            collo_actors = vedo.Points(collocation_points, r=4)
            self._s["server_plot_actors"]["collocation_points"] = collo_actors

            self.show(
                collo_actors,
                at=self._c["server_plot"],
                mode=self._c["plotter_mode"],
            )
            # counter++
            # this should be from 1 to 2
            self._s["right_click_counter"] += 1
            return None

        # remove if right click counter is not 1
        # this also ensures that continuous right click works
        # although, it will recompute each time
        server_actors = self._s.get("server_plot_actors", None)
        if server_actors is not None:
            self.remove(*server_actors.values(), at=self._c["server_plot"])
            # reset counter
            self._s["right_click_counter"] = 0

        # if there's server actors, there will be error actors
        # we can pack this into the `if` above also
        error_actors = self._s.get("error_plot_actors", None)
        if error_actors is not None:
            self.remove(*error_actors.values(), at=self._c["error_plot"])

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

            # use custom refine function to refine
            self._c["refine"](solution)

            self._logd(
                f"created solution spline - degrees: {solution.ds}"
                f"knot_vectors: {solution.kvs}"
            )

            # get greville points and sparsity pattern
            # use custom collocate function
            queries = self._c["collocate"](self._s["solution_spline"])
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

        # plot error
        err_actor = self._s["spline_actors"]["spline"].clone()
        u_q = splinepy.utils.data.uniform_query(
            solution.parametric_bounds,
            splinepy.utils.data._enforce_len(
                self._c["sample_resolutions"], self._s["spline"].para_dim
            ),
        )
        err = mapper.laplacian(u_q)
        err = np.log10(abs(err + 1))

        worst_id = np.argmax(err)
        w_point = vedo.Point(self._s["spline"].evaluate([u_q[worst_id]])[0])

        err_actor.cmap("coolwarm", err).add_scalarbar3d()
        self._s["error_plot_actors"] = {
            "spline": err_actor,
            "knots": self._s["spline_actors"]["knots"],
            "worst": w_point,
        }
        self.show(
            *self._s["error_plot_actors"].values(),
            at=self._c["error_plot"],
            mode=self._c["plotter_mode"],
        )

        # counter += 1
        # this should only set 0 to 1
        # otherwise something went wrong.
        self._s["right_click_counter"] += 1
        assert self._s["right_click_counter"] == 1

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
            "Solution - right click to sync\n& view collocation points",
            at=self._c["server_plot"],
            interactive=False,
            mode=self._c["plotter_mode"],
        )

        self.show(
            "Error - red dot marks the maximum",
            at=self._c["error_plot"],
            interactive=False,
            mode=self._c["plotter_mode"],
        )

        # let's start
        self.interactive()

        return self
