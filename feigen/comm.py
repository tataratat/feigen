"""
Communication helpers.
"""
import uuid

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from feigen import log
from feigen._base import FeigenBase


def new_uuid():
    """
    Returns a new uuid in str.

    Parameters
    ----------
    None

    Returns
    -------
    uuid_str: str
    """
    return str(uuid.uuid4())


def to_str(dict_):
    """Given dict, prepares a str that's readable from iganet server.

    Parameters
    ----------
    dict_: dict
      json request. No further conversion is required as long as all the values
      are python native type

    Returns
    -------
    sendable_str: str
      string encoded in bytes.
    """
    return str(dict_).replace("'", '"').encode()


def has_same_id(request_form, response_form, raise_=True, message=None):
    """
    Checks if request form and response form has the same
    id.

    Parameters
    ----------
    request_form: dict
    response_form: dict

    Returns
    -------
    has_same_id: bool
    """
    if request_form["id"].startswith(response_form["request"]):
        # matches. return True
        return True

    # else:

    msg = (
        f"id mismatch between request ({request_form['id']}) "
        f" and response ({response_form['request']})!"
    )
    if message is not None:
        msg = " -- ".join([message, msg])

    if raise_:
        raise ValueError(msg)

    # warning log
    log.warning(msg)

    return False


def error_check(response_form):
    """
    Checks status and raises RuntimeError in case comm failed.

    Parameters
    ----------
    response_form: dict

    Returns
    -------
    None


    Raises
    ------
    RuntimeError
    """
    if response_form["status"] != 0:
        raise RuntimeError(
            "Error occurred during communication - "
            f"{response_form['status']}"
        )


def iganet_to_splinepy(data_dict):
    """
    Creates spline based on iganet's response dict.

    Parameters
    ----------
    data_dict: dict

    Returns
    -------
    splinepy_bspline: splinepy.BSpline
    """
    iganet_spline = data_dict["data"]

    for_splinepy = {}

    # degrees - same format
    for_splinepy["degrees"] = iganet_spline["degrees"]

    # coeffs (cps) are row major. So reorganize that
    coeffs = iganet_spline["coeffs"]
    for_splinepy["control_points"] = [[*x] for x in zip(*coeffs)]

    # knot_vectors - also the same
    for_splinepy["knot_vectors"] = iganet_spline["knots"]

    return for_splinepy


def recv_until_id_matches(request_form, message, max_recv=10):
    """
    Returns a function that calls recv until id matches.
    This is useful in case there are broadcasts
    in between your communication.

    Parameters
    ----------
    request_form: dict
    max_recv: int
      Number of maximum recv call

    Returns
    -------
    func: function
      f(recv) -> eval(response)
    """

    def func(recv):
        for _ in range(max_recv):
            response = eval(recv())
            if has_same_id(
                request_form,
                response,
                False,
                "/".join(["recv_until_id_matches()", message]),
            ):
                return response

    return func


class RequestForm(FeigenBase):
    """
    Collection of request forms that follows the protocol.
    Session and spline independent requests are available
    as class methods.

    Parameters
    ----------
    session_id: str
    spline_id: str
    """

    __slots__ = ("_ws", "_session_id")

    @classmethod
    def template(cls):
        """
        Request form, with pre-written id (uuid)
        """
        return {"id": new_uuid()}

    @classmethod
    def create_session(cls, websocket):
        """
        Creates session at given websocket

        Parameters
        ----------
        websocket: WebSocketClient

        Returns
        -------
        session_id: str
         uuid
        """
        req = cls.template()
        req["request"] = "create/session"

        recv = websocket.send_recv(to_str(req), eval_=True)

        has_same_id(req, recv, raise_=True, message="create_session()")
        error_check(recv)

        return recv["data"]["id"]

    def __init__(self, websocket, session_id):
        """
        You can also keep the form alive for get / put options.
        As one session can hold more than one spline, each
        request form has an option to specify model id
        """
        if not isinstance(websocket, WebSocketClient):
            raise TypeError(
                f"Invalid websocket type {type(websocket)}."
                "Expects WebSocketClient."
            )

        self._ws = websocket
        self._session_id = str(session_id)

    def create_spline(self, model_name, degree=None, ncoeffs=None):
        """
        Create specified spline and returns its id.

        Parameters
        ----------
        model_name: str

        Returns
        -------
        model_id: int
        """
        req = self.template()
        req["request"] = f"create/{self._session_id}/{model_name}"
        req["data"] = {}

        if degree is not None:
            req["data"]["degree"] = degree
        if ncoeffs is not None:
            req["data"]["ncoeffs"] = ncoeffs

        self._logd(f"create_spline request - {req}")

        recv = self._ws.send_recv(to_str(req), eval_=True)
        has_same_id(req, recv, raise_=True, message="create_spline()")
        error_check(recv)

        return recv["data"]["id"]

    def model_info(self, model_id, to_splinepy_dict=True):
        """
        Returns model information. If to_splinepy_dict==True,
        you can use the return value to init splinepy's spline.

        Parameters
        ----------
        model_id: int
        to_splinepy_dict: bool
          Default is True

        Returns
        -------
        response: dict
        """
        req = self.template()
        req["request"] = f"get/{self._session_id}/{model_id}"

        # currently only configured for bspline, but
        # here could be a place for a switch among model types
        server_spline = self._ws.send_recv(to_str(req), eval_=True)
        has_same_id(req, server_spline, raise_=True, message="model_info()")
        error_check(server_spline)

        if to_splinepy_dict:
            return iganet_to_splinepy(server_spline)

        # else
        return server_spline

    def sync_coeffs(self, model_id, coeffs):
        """
        Syncs iganet model using sync_from.
        Server response from this request includes some broadcasting

        Parameters
        ----------
        model_id: int
        coeffs: iterable

        Returns
        -------
        response: dict
        """
        req = self.template()

        # get all coeffs
        req["request"] = f"put/{self._session_id}/{model_id}/coeffs"
        req["data"] = {
            "indices": list(range(len(coeffs))),
            "coeffs": coeffs.tolist(),
        }

        synced = self._ws.send_recv(
            to_str(req),
            eval_=True,
            recv_hook=recv_until_id_matches(req, "sync_coeffs()"),
        )

        has_same_id(req, synced, raise_=True, message="sync_coeffs()")
        error_check(synced)

        return synced

    def evaluate_model(self, model_id, component, resolution):
        """
        Evaluates model.

        Parameters
        ----------
        model_id: int
        component: str
          For example {"ValueFieldMagnitude", "ValueField", ...}
        resolution: list
          probably a good idea to enforce_len before

        Returns
        -------
        evaluated: list
          list of evaluated points
        """
        req = self.template()

        req["request"] = f"eval/{self._session_id}/{model_id}/{component}"
        req["data"] = {"resolution": resolution}

        evaluated = self._ws.send_recv(
            to_str(req),
            eval_=True,
            recv_hook=recv_until_id_matches(req, "evaluate_model()"),
        )
        has_same_id(req, evaluated, raise_=True, message="evaluate_model()")
        error_check(evaluated)

        return evaluated["data"]


class WebSocketClient(FeigenBase):
    """
    Minimal websocket client using `websockets`'s thread
    based implementation.
    Can be used to send and receive data from a websocket
    server.

    Examples
    --------
    >>> websocket = feigen.comm.WebSocketClient("ws://localhost:1111")
    >>> send_message = "Field data of mesh number 5, please."
    >>> websocket.send_recv(send_message)
    [1, 1, 2, 3, 5]
    """

    def __init__(self, uri, close_timeout=60):
        """
        Parameters
        ----------
        uri: str
        close_timeout: int
        """
        # save config in case we need to re connect
        self.uri = uri
        self.close_timeout = close_timeout
        self.websocket = connect(uri, close_timeout=close_timeout)

    def send_recv(self, message, eval_=True, recv_hook=None):
        """
        Send message and return received answer.

        Parameters
        ----------
        message: str
        eval_: bool
          calls eval() on received message before returning.
          This is only relevant iff recv_hook is None.
        recv_hook: callable
          Hook that takes recv function as an argument.

        Returns
        -------
        response: Any
          str, unless recv_hook returns otherwise.
        """
        try:
            self.websocket.ping()
        except (ConnectionClosed, RuntimeError):
            self._logd("connection error. trying to reconnect.")
            # try to reconnect
            self.websocket = connect(
                self.uri, close_timeout=self.close_timeout
            )

        self.websocket.send(message)

        if recv_hook is None:
            recv = self.websocket.recv()

            # eval() recv. should be a dict
            if eval_:
                return eval(recv)

            return recv

        else:
            return recv_hook(self.websocket.recv)
