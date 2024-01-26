import torch
from torch import nn, Tensor


# time is 1D tensor as required
# u is in a different format: torchcde wants (batch, time, dof), ours is (batch, dof, time)
# Later when we implement different interpolation schemes, may need to define an InterpolationBase class
# and set up a class hierarchy
def divmod_time(
    t: Tensor,
    t_anchor: Tensor,
) -> torch.LongTensor:
    inds = torch.searchsorted(t_anchor, t, right=True) - 1
    return inds


def calculate_cubic_smoothing_spline_coeffs(
    ts: Tensor, us: Tensor, alpha: float = 0.0
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor,]:
    """
    Computes the coefficients for a batch of cubic smoothing splines on data trajectories generated from a function F : R -> R^M

    Parameters
    ----------
    ts : torch.Tensor
            Sampled time points to compute the smoothing splines on
            1D Tensor of size (N, ); N = number of points per trajectory
    us : torch.Tensor
        Corresponding sampled data values to compute the smoothing splines on
        3D Tensor of size (B, M, N); B = number of trajectories, M = data dimensionality, N = number of points per trajectory
    alpha : float
        Smoothing parameter. Must be non-negative. alpha = 0 corresponds to no smoothing (interpolation).

    Returns
    -------
    ts : torch.Tensor
        Sampled time points to compute the smoothing splines on
            1D Tensor of size (N, ); N = number of points per trajectory
    a : torch.Tensor
        Coefficient of the cubic polynomial at^3+bt^2+ct+d.
        3D Tensor of size (B, M, N)
    b : torch.Tensor
        Coefficient of the cubic polynomial at^3+bt^2+ct+d.
        3D Tensor of size (B, M, N)
    c : torch.Tensor
        Coefficient of the cubic polynomial at^3+bt^2+ct+d.
        3D Tensor of size (B, M, N)
    d : torch.Tensor
        Coefficient of the cubic polynomial at^3+bt^2+ct+d.
        3D Tensor of size (B, M, N)
    """
    h = ts[1:] - ts[:-1]
    _Q = _make_Q_matrix(h)
    _R = _make_R_matrix(h)
    g_ddot, g = _make_spline_coeffs(us, h, _Q, _R, alpha)
    a, b, c, d = _to_polynomial_coeffs(h, g, g_ddot)
    return ts, a, b, c, d


def _make_spline_coeffs(us, h, Q, R, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the cubic smoothing spline coefficients g_ddot and g for a given value of the smoothing parameter.
    Parameters
    ----------
    alpha : float
        Smoothing parameter. Must be non-negative. alpha = 0 corresponds to no smoothing (interpolation).
    Returns
    -------
    g_ddot : torch.Tensor
        Second derivative of the smoothing spline at the knot points.
        3D Tensor of size (B, M, N)
    g : torch.Tensor
        Function value of the smoothing spline at the knot points.
        3D Tensor of size (B, M, N)
    """

    with torch.no_grad():
        u2, u1, u0 = (
            us[..., 2:],
            us[..., 1:-1],
            us[..., 0:-2],
        )
        QT_Y = (u2 - u1) / h[1:] - (u1 - u0) / h[0:-1]
        B = torch.addmm(R, Q.T, Q, alpha=alpha)

        out = torch.linalg.ldl_factor(B)
        g_ddot = torch.linalg.ldl_solve(
            *out, torch.unsqueeze(QT_Y, dim=-1)
        )  # (B, M, N-2, 1)-
        g = us - alpha * (Q @ g_ddot).squeeze(-1)
        g_ddot = nn.functional.pad(g_ddot.squeeze(-1), (1, 1), "constant", 0.0)

    return g_ddot, g


def _to_polynomial_coeffs(
    h: Tensor,
    g: Tensor,
    g_ddot: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor,]:
    a, b, c = torch.zeros_like(g), torch.zeros_like(g), torch.zeros_like(g)
    a[..., :-1] = (g_ddot[..., 1:] - g_ddot[..., :-1]) / (6 * h)
    b[..., :-1] = g_ddot[..., :-1] / 2
    c[..., :-1] = (g[..., 1:] - g[..., :-1]) / h - (1 / 6) * h * (
        g_ddot[..., 1:] + 2 * g_ddot[..., :-1]
    )
    d = g
    return a, b, c, d


class PiecewiseLinear(nn.Module):
    def __init__(self, ts: Tensor, us: Tensor):
        super().__init__()
        self.register_buffer("ts", ts.detach().clone())
        self.register_buffer("us", us.detach().clone())


class CubicSplineSmoothing(nn.Module):
    """
    A class to compute a batch of cubic smoothing splines on data trajectories generated from a function F : R -> R^M

    ...

    Attributes
    ----------
    ts : torch.Tensor
        Sampled time points (knot points) to compute the smoothing splines on.
        1D Tensor of size (N, ); N = number of points per trajectory
    us : torch.Tensor
        Corresponding sampled data values to compute the smoothing splines on.
        3D Tensor of size (B, M, N); B = number of trajectories, M = data dimensionality, N = number of points per trajectory
    g_ddot : torch.Tensor
        Second derivative of the smoothing spline at the knot points.
        3D Tensor of size (B, M, N)
    g : torch.Tensor
        Function value of the smoothing spline at the knot points.
        3D Tensor of size (B, M, N)
    _h : torch.Tensor
        Time intervals between subsequent points in the trajectory.
        2D Tensor of size (N-1, )
    _Q : torch.Tensor
        Q matrix for the spline coefficient calculation
        2D Tensor of size (N, N-2)
    _R : torch.Tensor
        R matrix for the spline coefficient calculation
        2D Tensor of size (N-2, N-2)
    """

    def __init__(self, ts: torch.Tensor, us: torch.Tensor, alpha: float = 0.0):
        """
        Parameters
        ----------
        ts : torch.Tensor
            Sampled time points to compute the smoothing splines on
            1D Tensor of size (N, ); N = number of points per trajectory
        us : torch.Tensor
            Corresponding sampled data values to compute the smoothing splines on
            3D Tensor of size (B, M, N); B = number of trajectories, M = data dimensionality, N = number of points per trajectory
        alpha : float
            Smoothing parameter. Must be non-negative. alpha = 0 corresponds to no smoothing (interpolation).
        """
        super().__init__()
        self.register_buffer("ts", ts.detach().clone())
        self.register_buffer("us", us.detach().clone())
        self._alpha = float(alpha)

        self.register_buffer("_h", self.ts[1:] - self.ts[:-1])
        self.register_buffer("_Q", _make_Q_matrix(self._h))
        self.register_buffer("_R", _make_R_matrix(self._h))

        g_ddot, g = self.make_spline_coeffs(alpha=self.alpha)
        self.register_buffer("g_ddot", g_ddot)
        self.register_buffer("g", g)

    @property
    def alpha(self) -> float:
        """
        Get or set the non-negative smoothing parameter alpha. alpha = 0 corresponds to no smoothing (interpolation).
        Trying to set alpha to a negative value will result in a ValueError
        Setting alpha also automatically updates the spline coefficients

        Returns
        -------
        alpha : float
            Smoothing parameter. Is a non-negative number.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0:
            raise ValueError("Smoothing parameter alpha must be non-negative!")
        g_ddot, g = self.make_spline_coeffs(value)
        self.register_buffer("g_ddot", g_ddot)
        self.register_buffer("g", g)
        self._alpha = value

    def make_spline_coeffs(self, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the cubic smoothing spline coefficients g_ddot and g for a given value of the smoothing parameter.

        Parameters
        ----------
        alpha : float
            Smoothing parameter. Must be non-negative. alpha = 0 corresponds to no smoothing (interpolation).

        Returns
        -------
        g_ddot : torch.Tensor
            Second derivative of the smoothing spline at the knot points.
            3D Tensor of size (B, M, N)
        g : torch.Tensor
            Function value of the smoothing spline at the knot points.
            3D Tensor of size (B, M, N)
        """

        with torch.no_grad():
            Q, R = self._Q, self._R
            u2, u1, u0 = (
                self.us[..., 2:],
                self.us[..., 1:-1],
                self.us[..., 0:-2],
            )
            QT_Y = (u2 - u1) / self._h[1:] - (u1 - u0) / self._h[0:-1]
            B = torch.addmm(R, Q.T, Q, alpha=alpha)

            out = torch.linalg.ldl_factor(B)
            g_ddot = torch.linalg.ldl_solve(
                *out, torch.unsqueeze(QT_Y, dim=-1)
            )  # (B, M, N-2, 1)
            g = self.us - alpha * (Q @ g_ddot).squeeze(-1)
            g_ddot = nn.functional.pad(g_ddot.squeeze(-1), (1, 1), "constant", 0.0)

        return g_ddot, g

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the values of the cubic smoothing spline for given time points.

        Parameters
        ----------
        t : torch.Tensor
            Time points to compute the cubic smoothing spline values on.
            1D Tensor with shape (K, ); K = number of query time points.

        Returns
        -------
        u_smoothed : torch.Tensor
            Computed cubic smoothing spline values for the given time points.
            3D Tensor with shape (B, M, K); B = batch size of the cubic interpolant, M = data dimensionality, K = number of query time points.
        """
        t = t.view(-1)
        with torch.no_grad():
            K = t.size(0)
            inds = get_interval_inds(t, self.ts)
            h, t_query, g_ddot, g = self.get_interval_coeffs(inds)

        Dt_l, Dt_r = t - t_query[:K], t_query[K:] - t
        Dt_h_l, Dt_h_r = Dt_l / h, Dt_r / h
        u_smoothed = (Dt_h_l * g[..., K:] + Dt_h_r * g[..., :K]) - Dt_l * Dt_r * (
            (1 + Dt_h_l) * g_ddot[..., K:] + (1 + Dt_h_r) * g_ddot[..., :K]
        ) / 6
        return u_smoothed

    def get_interval_coeffs(
        self, inds: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fetches the necessary coefficients required for the cubic smoothing spline evaluation for interval indices.
        Note that the coefficients for t, g_ddot, and g are fetched concatenated more computational efficiency.

        Parameters
        ----------
        inds : torch.LongTensor
            Indices of the intervals for which the corresponding coefficients have to be fetched.
            1D LongTensor with shape (K, ); K = number of query time points.

        Returns
        -------
        h_query : torch.Tensor
            Time interval sizes for the given indices
            1D Tensor with shape (K, ); K = number of query time points.
        t_query : torch.Tensor
            Concatenated left and right time points corresponding to the given indices.
            The first K elements correspond to the left values, the next K elements correspond to the right.
            1D Tensor with shape (2K, )
        g_ddot_query : torch.Tensor
            Concatenated left and right g_ddot coefficients corresponding to the given indices.
            The first K elements along the last dimension correspond to the left values, the next K elements correspond to the right.
            3D Tensor with shape (B, M, 2K); M = dimensionality of the data.
        g_query : torch.Tensor
            Concatenated left and right g coefficients corresponding to the given indices.
            The first K elements along the last dimension correspond to the left values, the next K elements correspond to the right.
            3D Tensor with shape (B, M, 2K).
        """
        h_query = torch.index_select(self._h, -1, inds)
        inds_total = torch.cat((inds, inds + 1), dim=-1)
        t_query = torch.index_select(self.ts, -1, inds_total)
        g_ddot_query = torch.index_select(self.g_ddot, -1, inds_total)
        g_query = torch.index_select(self.g, -1, inds_total)
        return h_query, t_query, g_ddot_query, g_query


def _make_Q_matrix(h: torch.Tensor) -> torch.Tensor:
    """
    Computes the Q matrix used to calculate the smoothing spline coefficients g_ddot and g.

    Parameters
    ----------
    h : torch.Tensor
        Time intervals between subsequent points in the trajectory.
        1 Tensor of size (N-1, ); N = number of points per trajectory

    Returns
    -------
    Q : torch.Tensor
        Q matrix for the spline coefficient calculation
        2D Tensor of size (N, N-2); N = number of points per trajectory
    """
    h_left, h_right = h[:-1], h[1:]
    Q_diag, Q_offdiag2 = 1 / h_left, 1 / h_right
    Q_offdiag1 = -Q_diag - Q_offdiag2

    N_1 = h.size(0)
    Q = torch.zeros((N_1 + 1, N_1 - 1), dtype=h.dtype)
    Q = torch.diagonal_scatter(Q, Q_diag, offset=0)
    Q = torch.diagonal_scatter(Q, Q_offdiag2, offset=-2)
    Q = torch.diagonal_scatter(Q, Q_offdiag1, offset=-1)
    return Q


def _make_R_matrix(h: torch.Tensor) -> torch.Tensor:
    """
    Computes the R matrix used to calculate the smoothing spline coefficients g_ddot and g.

    Parameters
    ----------
    h : torch.Tensor
        Time intervals between subsequent points in the trajectory.
        1D Tensor of size (N-1, ); N = number of points per trajectory

    Returns
    -------
    R : torch.Tensor
        R matrix for the spline coefficient calculation
        2D Tensor of size (N-2, N-2); N = number of points per trajectory
    """
    h_left, h_right = h[:-1], h[1:]
    R_diag = (h_left + h_right) / 3
    R_offdiag = h_right[:-1] / 6

    N_1 = h.size(0)
    R = torch.zeros((N_1 - 1, N_1 - 1), dtype=h.dtype)
    R = torch.diagonal_scatter(R, R_diag, offset=0)
    R = torch.diagonal_scatter(R, R_offdiag, offset=1)
    R = torch.diagonal_scatter(R, R_offdiag, offset=-1)
    return R


# @torch.jit.script
def get_interval_inds(t: torch.Tensor, t_grid: torch.Tensor) -> torch.LongTensor:
    """
    Given time points t and a monotonically increasing 1D sequence of reference timepoints,
    compute the index of the bin that the time points belong in.

    Parameters
    ----------
    t : torch.Tensor
        Time points to query the bin indices of
        1D Tensor of size (K, ); K = number of time points to be queried
    t_grid: torch.Tensor
        Reference timepoints for querying. Must be monotonically increasing.
        1D Tensor of size (N, ); N = number of reference time points

    Returns
    -------
    inds : torch.LongTensor
        Indices of the intervals each of the input time points belong in.
        1D LongTensor of size (K, )
    """
    inds = torch.bucketize(t, t_grid, right=True) - 1
    return torch.clamp(inds, 0, len(t_grid) - 2)


def get_interval_inds_batched(
    t: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.LongTensor:
    """
    Batched version of get_interval_inds.
    The operation is performed along the first axis, which is considered to be the batch dimension.

    Parameters
    ----------
    t : torch.Tensor
        Batch of time points to query the bin indices of.
        2D Tensor of size (B, K); B = batch size, K = number of time points to be queried per batch.
    t_grid: torch.Tensor
        Batch of reference timepoints for querying. Must be monotonically increasing along the second dimension.
        2D Tensor of size (B, N); B = batch size, N = number of reference time points per batch.

    Returns
    -------
    inds : torch.LongTensor
        Indices of the intervals each of the input time points belong in for each batch.
        2D LongTensor of size (B, K); B = batch size, K = number of time points to be queried per batch.
    """
    inds = torch.stack(
        [get_interval_inds(t_i, t_grid_i) for t_grid_i, t_i in zip(t_grid, t)], dim=0
    )
    return inds


@torch.jit.script
def index_select_batched(A: torch.Tensor, inds: torch.LongTensor):
    """
    Batched version of torch.index_select. Batch dimension is considered to be the first dimension.
    Index selecting is performed along the last dimension.

    Parameters
    ----------
    A : torch.Tensor
        Tensor to select elements from.
        n(>=2)D Tensor of size (B, ..., N); B = batch size, N = number of elements in the index selecting dimension.
    inds : torch.LongTensor
        Batch of indices to select elements of A with.
        2D LongTensor of size (B, K); B = batch size, K = number of elements to be index-selected.

    Returns
    -------
    A_selected : torch.Tensor
        Tensor of selected elements of A.
        n(>=2)D Tensor of size (B, ..., K); B = batch size, K = number indeices used for selection.
    """
    return torch.stack(
        [torch.index_select(A_, -1, inds_) for A_, inds_ in zip(A, inds)], dim=0
    )


INTERPOLATION_DICT = {"cubic": CubicSplineSmoothing}


def get_interpolation(name: str) -> CubicSplineSmoothing:
    try:
        interpolant = INTERPOLATION_DICT[name]
    except KeyError:
        raise ValueError(
            f"Interpolant name not recognized. Must be one of {INTERPOLATION_DICT.keys()}"
        )
    return interpolant
