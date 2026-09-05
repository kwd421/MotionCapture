"""Explicit terminal errors for the prototype user flow."""


class PrototypeError(RuntimeError):
    """Base class for an observable prototype failure."""


class ModelAssetError(PrototypeError):
    """The pinned model asset is missing or invalid."""


class AvatarAssetError(PrototypeError):
    """A pinned third-party avatar asset is missing, invalid, or incomplete."""


class CameraError(PrototypeError):
    """The explicitly selected camera could not satisfy the capture contract."""


class InferenceError(PrototypeError):
    """Holistic inference failed for the current frame."""


class SessionRecordError(PrototypeError):
    """The metadata-only session record could not be written."""
