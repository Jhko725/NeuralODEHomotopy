# ruff: noqa: F722
import torch
from torch import Tensor, nn
from jaxtyping import Float, Integer


class CubicSplineSmoothing(nn.Module):
    """A class to compute a batch of cubic smoothing splines on data trajectories generated from a function F : R -> R^M

    Can accept a batch of multi-dimensional trajectories. However, the individual trajectories in the batch must share the same time points.
    The implementation is based on the following reference:
    Green, Peter J., and Bernard W. Silverman. Nonparametric Regression and Generalized Linear Models: A Roughness Penalty Approach.
    1st edition. Monographs on Statistics and Applied Probability 58. Chapman & Hall /CRC, 2000.

    Attributes:
        ts: Tensor of sampled time points (knot points) to compute the smoothing splines on.
            Must be monotonically increasing. Has shape (N, ); N = number of sampled time points.
        us: Tensor of sampled data values to compute the smoothing splines on.
            Has shape (B, M, N); B = number of trajectories, M = data dimensionality, N = number of sampled time points.
        g_ddot: Tensor of the second derivatives of the smoothing spline at the knot points.
            Has shape (B, M, N).
        g: Tensor of the function values of the smoothing spline at the knot points.
           Has shape (B, M, N).
        n_batch: Number of trajectories the interpolant is constructed on; i.e. B.
        n_dim: Number of dimensions a single trajectory of the interpolant has; i.e. M.
    """

    def __init__(
        self,
        ts: Float[Tensor, " N"],
        us: Float[Tensor, "B M N"],
        alpha: float = 0.0,
    ):
        """Initializes the class by constructing the smoothing cubic spline.

        Args:
            ts: Tensor of sampled time points to compute the smoothing splines on.
                Has shape (N, ); N = number of sampled time points.
            us: Tensor of sampled data values to compute the smoothing splines on.
                Has shape (B, M, N); B = number of trajectories, M = data dimensionality, N = number of sampled time points.
            alpha: A non-negative scalar denoting the smoothing parameter.
                alpha = 0 corresponds to no smoothing (interpolation). Passing a negative value will result in a ValueError.
        """
        super().__init__()
        self.register_buffer("ts", ts.detach().clone())
        self.register_buffer("us", us.detach().clone())
        self.alpha = float(alpha)

        g_ddot, g = self.make_spline_coeffs(alpha=self.alpha)
        self.register_buffer("g_ddot", g_ddot)
        self.register_buffer("g", g)

    @property
    def n_batch(self) -> int:
        """Number of trajectories the interpolant is constructed on.

        Returns:
            n_batch: Integer corresponding to the number of trajectories underlying the interpolant.
        """
        return self.us.shape[0]

    @property
    def n_dim(self) -> int:
        """Number of dimensions a single trajectory of the interpolant has.

        Returns:
            n_dim: Integer corresponding to the number of dimensions a trajectory of the interpolant has.
        """
        return self.us.shape[1]

    @property
    def alpha(self) -> float:
        """A non-negative scalar denoting the smoothing parameter.

        alpha = 0 corresponds to no smoothing (interpolation). Passing a negative value will result in a ValueError.
        Modifying alpha after class instantiation will automatically update the spline coefficients.

        Returns:
            alpha : A non-negative float corresponding to the smoothing parameter.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha_new: float) -> None:
        if alpha_new < 0:
            raise ValueError("Smoothing parameter alpha must be non-negative!")
        g_ddot, g = self.make_spline_coeffs(alpha_new)
        self.register_buffer("g_ddot", g_ddot)
        self.register_buffer("g", g)
        self._alpha = alpha_new

    @property
    def _h(self) -> Float[Tensor, " N-1"]:
        """A Tensor of time intervals between subsequent points in the trajectory.

        This is needed to calculate the spline coefficients.

        Returns:
            h: A Tensor of shape (N-1, ) containing the time intervals; N = number of sampled time points.
        """
        return self.ts[1:] - self.ts[:-1]

    def _make_Q_matrix(self) -> Float[Tensor, "N N-2"]:
        """Computes the Q matrix used to calculate the smoothing spline coefficients g_ddot and g.

        Returns:
            Q: A Tensor representing the Q matrix for the spline coefficient calculation.
            Has shape (N, N-2); N = number of points per trajectory.
        """
        h = self._h
        h_left, h_right = h[:-1], h[1:]
        Q_diag, Q_offdiag2 = 1 / h_left, 1 / h_right
        Q_offdiag1 = -Q_diag - Q_offdiag2

        N_1 = h.size(0)
        Q = torch.zeros((N_1 + 1, N_1 - 1), dtype=h.dtype)
        Q = torch.diagonal_scatter(Q, Q_diag, offset=0)
        Q = torch.diagonal_scatter(Q, Q_offdiag2, offset=-2)
        Q = torch.diagonal_scatter(Q, Q_offdiag1, offset=-1)
        return Q

    def _make_R_matrix(self) -> Float[Tensor, "N-2 N-2"]:
        """Computes the R matrix used to calculate the smoothing spline coefficients g_ddot and g.

        Returns:
            R: A Tensor representing the R matrix for the spline coefficient calculation.
            Has shape (N-2, N-2); N = number of points per trajectory.
        """
        h = self._h
        h_left, h_right = h[:-1], h[1:]
        R_diag = (h_left + h_right) / 3
        R_offdiag = h_right[:-1] / 6

        N_1 = h.size(0)
        R = torch.zeros((N_1 - 1, N_1 - 1), dtype=h.dtype)
        R = torch.diagonal_scatter(R, R_diag, offset=0)
        R = torch.diagonal_scatter(R, R_offdiag, offset=1)
        R = torch.diagonal_scatter(R, R_offdiag, offset=-1)
        return R

    def make_spline_coeffs(
        self, alpha: float
    ) -> tuple[
        Float[Tensor, "B M N"],
        Float[Tensor, "B M N"],
    ]:
        """Calculate the cubic smoothing spline coefficients g_ddot and g for a given value of the smoothing parameter.

        Note that the alpha value used to calculate the coefficients is not referenced from self.alpha,
        but rather passed in as the function argument.
        This was deliberately done to make the internals of the setter method for alpha more explicit.
        However, this also means that the function does not check whether the given alpha value is non-negative.

        Args:
            alpha: A non-negative float value denoting the smoothing parameter.
            alpha = 0 corresponds to no smoothing (interpolation).

        Returns:
            g_ddot: Tensor of the second derivatives of the smoothing spline at the knot points.
                Has shape (B, M, N); B = number of trajectories, M = data dimensionality, N = number of points per trajectory.
            g: Tensor of the function values of the smoothing spline at the knot points.
                Has shape (B, M, N).
        """

        with torch.no_grad():
            Q = self._make_Q_matrix()
            R = self._make_R_matrix()
            u2: Float[Tensor, "B M N-2"] = self.us[..., 2:]
            u1: Float[Tensor, "B M N-2"] = self.us[..., 1:-1]
            u0: Float[Tensor, "B M N-2"] = self.us[..., 0:-2]

            QT_Y = (u2 - u1) / self._h[1:] - (u1 - u0) / self._h[0:-1]
            B = torch.addmm(R, Q.T, Q, alpha=alpha)

            out = torch.linalg.ldl_factor(B)
            g_ddot: Float[Tensor, "B M N-2 1"] = torch.linalg.ldl_solve(
                *out, torch.unsqueeze(QT_Y, dim=-1)
            )

            g: Float[Tensor, "B M N"] = self.us - alpha * (Q @ g_ddot).squeeze(-1)
            g_ddot: Float[Tensor, "B M N"] = nn.functional.pad(
                g_ddot.squeeze(-1), (1, 1), "constant", 0.0
            )

        return g_ddot, g

    def forward(self, t: Float[Tensor, " K"]) -> Float[Tensor, "B M K"]:
        """Computes the values of the cubic smoothing spline for given time points.

        Args:
            t: Tensor of time points to compute the cubic smoothing spline values on.
            Has shape (K, ); K = number of query time points.

        Returns:
            u_smoothed: Tensor of computed cubic smoothing spline values for the given time points.
            Has shape (B, M, K); B = batch size of the cubic interpolant, M = data dimensionality, K = number of query time points.
        """
        t = t.view(-1)
        with torch.no_grad():
            K = t.size(0)
            inds = self.get_interval_inds(t)
            h, t_query, g_ddot, g = self.get_interval_coeffs(inds)

        Dt_l, Dt_r = t - t_query[:K], t_query[K:] - t
        Dt_h_l, Dt_h_r = Dt_l / h, Dt_r / h
        u_smoothed = (Dt_h_l * g[..., K:] + Dt_h_r * g[..., :K]) - Dt_l * Dt_r * (
            (1 + Dt_h_l) * g_ddot[..., K:] + (1 + Dt_h_r) * g_ddot[..., :K]
        ) / 6
        return u_smoothed

    def get_interval_inds(self, t: Float[Tensor, " K"]) -> Integer[Tensor, " K"]:
        """Given time points t, compute the index of the bin that the time points belong in,
        with respect to the monotonically increasing 1D sequence of reference timepoints self.ts.

        Args:
            t: Tensor of time points to query the bin indices of
            Has shape (K, ); K = number of time points to be queried

        Returns:
            inds: A tensor of indices indices of the intervals each of the input time point belong in.
            Has shape (K, )
        """
        inds = torch.bucketize(t, self.ts, right=True) - 1
        return torch.clamp(inds, 0, len(self.ts) - 2)

    def get_interval_coeffs(
        self, inds: Integer[Tensor, " K"]
    ) -> tuple[
        Float[Tensor, " K"],
        Float[Tensor, " 2*K"],
        Float[Tensor, "B M 2*K"],
        Float[Tensor, "B M 2*K"],
    ]:
        """Fetches the necessary coefficients required for the cubic smoothing spline evaluation for interval indices.

        Note that the coefficients for t, g_ddot, and g are fetched concatenated to reduce the number of calls to torch.index_select().

        Args:
            inds: Tensor containing the indices of the intervals for which the corresponding coefficients have to be fetched.
            Has shape (K, ); K = number of query time points.

        Returns:
            h_query: Tensor containing time interval sizes for the given indices.
                Has shape (K, ); K = number of query time points.
            t_query: Tensor containning concatenated left and right time points corresponding to the given indices.
                The first K elements correspond to the left values, the next K elements correspond to the right.
                Has shape (2K, ).
            g_ddot_query: Tensor containing concatenated left and right g_ddot coefficients corresponding to the given indices.
                The first K elements along the last dimension correspond to the left values, the next K elements correspond to the right.
                Has shape (B, M, 2K); M = dimensionality of the data.
            g_query : Tensor containing concatenated left and right g coefficients corresponding to the given indices.
                The first K elements along the last dimension correspond to the left values, the next K elements correspond to the right.
                Has shape (B, M, 2K).
        """
        h_query = torch.index_select(self._h, -1, inds)
        inds_total = torch.cat((inds, inds + 1), dim=-1)
        t_query = torch.index_select(self.ts, -1, inds_total)
        g_ddot_query = torch.index_select(self.g_ddot, -1, inds_total)
        g_query = torch.index_select(self.g, -1, inds_total)
        return h_query, t_query, g_ddot_query, g_query

    def make_polynomial_coeffs(
        self,
    ) -> tuple[
        Float[Tensor, "B M N"],
        Float[Tensor, "B M N"],
        Float[Tensor, "B M N"],
        Float[Tensor, "B M N"],
    ]:
        """Converts the spline coefficients self.g_ddot and self.g into coeffcients of the piecewise cubic polynomial at^3+bt^2+ct+d.

        Returns:
            a: Tensor corresponding to the coefficent a.
                Has shape (B, M, N).
            b: Tensor corresponding to the coefficent b.
                Has shape (B, M, N).
            c: Tensor corresponding to the coefficent c.
                Has shape (B, M, N).
            d: Tensor corresponding to the coefficent d. Note that this is identical to self.g.
                Has shape (B, M, N).
        """
        h = self._h
        g: Float[Tensor, "B M N"] = self.g
        g_ddot: Float[Tensor, "B M N"] = self.g_ddot
        a, b, c = torch.zeros_like(g), torch.zeros_like(g), torch.zeros_like(g)

        a[..., :-1] = (g_ddot[..., 1:] - g_ddot[..., :-1]) / (6 * h)
        b[..., :-1] = g_ddot[..., :-1] / 2
        c[..., :-1] = (g[..., 1:] - g[..., :-1]) / h - (1 / 6) * h * (
            g_ddot[..., 1:] + 2 * g_ddot[..., :-1]
        )
        d = g
        return a, b, c, d
