"""RTMPose-M Halpe26 export view used by the TensorRT experiment.

The current PyTorch model enables flip-test inference.  A TensorRT engine
represents one forward pass, so the fast path explicitly disables flip test.
The runtime may later submit original and flipped crops as one batch and
average their SimCC vectors when accuracy is preferred over latency.
"""

_base_ = (
    "../../../assets/models/rtmpose/"
    "rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
)

model = dict(
    backbone=dict(init_cfg=None),
    test_cfg=dict(flip_test=False),
)
