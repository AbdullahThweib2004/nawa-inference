"""Round-trip and validation tests for nawa_format.py (numpy only, no torch)."""

import struct

import numpy as np
import pytest

import nawa_format as nf


def make_model(with_bias: bool = True) -> nf.Model:
    rng = np.random.default_rng(0)
    return nf.Model(
        input_shape=(4,),
        normalization=nf.Normalization(1 / 255, [0.1307], [0.3081]),
        layers=[
            nf.Linear(rng.standard_normal((4, 3), dtype=np.float32),
                      rng.standard_normal(3, dtype=np.float32) if with_bias else None),
            nf.ReLU(),
            nf.Sigmoid(),
            nf.Linear(rng.standard_normal((3, 2), dtype=np.float32), None),
            nf.Softmax(axis=-1),
        ],
    )


# ---------------------------------------------------------------------------
# Tensor blocks and files
# ---------------------------------------------------------------------------


def test_tensor_block_matches_spec_example():
    # The 2x3 example from docs/model_format.md, byte for byte.
    block = nf.encode_tensor_block(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    expected = bytes.fromhex(
        "02000000" "0200000000000000" "0300000000000000"
        "0000803f" "00000040" "00004040" "00008040" "0000a040" "0000c040"
    )
    assert block == expected


@pytest.mark.parametrize("shape", [(), (1,), (5,), (2, 3), (2, 3, 4), (1, 784)])
def test_tensor_file_round_trip(shape):
    array = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) - 3.5
    decoded = nf.decode_tensor_file(nf.encode_tensor_file(array))
    assert decoded.shape == shape
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, array)


def test_tensor_file_save_load(tmp_path):
    array = np.array([[0.25, -1.0], [3.0, 1e-7]], dtype=np.float32)
    nf.save_tensor(tmp_path / "t.ntsr", array)
    np.testing.assert_array_equal(nf.load_tensor(tmp_path / "t.ntsr"), array)


def test_float64_input_is_stored_as_float32():
    decoded = nf.decode_tensor_file(nf.encode_tensor_file(np.array([0.1, 0.2])))
    assert decoded.dtype == np.float32


def test_tensor_file_bad_magic():
    data = b"XXXX" + nf.encode_tensor_file(np.zeros(2))[4:]
    with pytest.raises(nf.NawaFormatError, match="magic"):
        nf.decode_tensor_file(data)


def test_tensor_file_bad_version():
    data = bytearray(nf.encode_tensor_file(np.zeros(2)))
    data[4:8] = struct.pack("<I", 99)
    with pytest.raises(nf.NawaFormatError, match="version"):
        nf.decode_tensor_file(bytes(data))


def test_tensor_file_truncated():
    data = nf.encode_tensor_file(np.zeros((2, 3)))
    with pytest.raises(nf.NawaFormatError, match="truncated"):
        nf.decode_tensor_file(data[:-1])


def test_tensor_file_trailing_bytes():
    with pytest.raises(nf.NawaFormatError, match="trailing"):
        nf.decode_tensor_file(nf.encode_tensor_file(np.zeros(2)) + b"\x00")


def test_zero_dimension_rejected():
    with pytest.raises(nf.NawaFormatError, match=">= 1"):
        nf.encode_tensor_block(np.zeros((2, 0)))
    # Hand-built block with dims {0}: the reader must reject it too.
    data = nf.TENSOR_MAGIC + struct.pack("<IIQ", 1, 1, 0)
    with pytest.raises(nf.NawaFormatError, match=">= 1"):
        nf.decode_tensor_file(data)


def test_too_many_dimensions_rejected():
    data = nf.TENSOR_MAGIC + struct.pack("<II", 1, 9)
    with pytest.raises(nf.NawaFormatError, match="ndim"):
        nf.decode_tensor_file(data)


# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_bias", [True, False])
def test_model_round_trip(with_bias):
    model = make_model(with_bias)
    decoded = nf.decode_model(nf.encode_model(model))

    assert decoded.input_shape == (4,)
    assert decoded.normalization.pixel_scale == pytest.approx(1 / 255)
    assert decoded.normalization.mean == pytest.approx([0.1307])
    assert decoded.normalization.std == pytest.approx([0.3081])
    assert [type(layer) for layer in decoded.layers] == [type(layer) for layer in model.layers]

    for original, loaded in zip(model.layers, decoded.layers):
        if isinstance(original, nf.Linear):
            np.testing.assert_array_equal(loaded.weight, original.weight)
            if original.bias is None:
                assert loaded.bias is None
            else:
                np.testing.assert_array_equal(loaded.bias, original.bias)
    assert decoded.layers[-1].axis == -1


def test_model_save_load(tmp_path):
    nf.save_model(tmp_path / "m.nawa", make_model())
    assert len(nf.load_model(tmp_path / "m.nawa").layers) == 5


def test_model_header_layout():
    data = nf.encode_model(make_model())
    assert data[:4] == b"NAWA"
    assert struct.unpack_from("<I", data, 4)[0] == 1          # version
    assert struct.unpack_from("<I", data, 8)[0] == 1          # input_ndim
    assert struct.unpack_from("<Q", data, 12)[0] == 4         # input_dims[0]
    assert struct.unpack_from("<I", data, 24)[0] == 1         # norm_count
    assert struct.unpack_from("<I", data, 36)[0] == 5         # num_layers
    assert struct.unpack_from("<I", data, 40)[0] == nf.LAYER_LINEAR


def test_model_bad_magic():
    data = b"NOPE" + nf.encode_model(make_model())[4:]
    with pytest.raises(nf.NawaFormatError, match="magic"):
        nf.decode_model(data)


def test_model_tensor_magic_is_not_a_model():
    with pytest.raises(nf.NawaFormatError, match="magic"):
        nf.decode_model(nf.encode_tensor_file(np.zeros(2)))


def test_model_bad_version():
    data = bytearray(nf.encode_model(make_model()))
    data[4:8] = struct.pack("<I", 2)
    with pytest.raises(nf.NawaFormatError, match="version"):
        nf.decode_model(bytes(data))


def test_model_unknown_layer_type():
    model = nf.Model((4,), nf.Normalization(1.0, [0.0], [1.0]), [nf.ReLU()])
    data = bytearray(nf.encode_model(model))
    data[-4:] = struct.pack("<I", 99)  # the ReLU type id is the last 4 bytes
    with pytest.raises(nf.NawaFormatError, match="unknown layer type id 99"):
        nf.decode_model(bytes(data))


def test_model_invalid_has_bias():
    model = nf.Model((2,), nf.Normalization(1.0, [0.0], [1.0]),
                     [nf.Linear(np.zeros((2, 2), dtype=np.float32))])
    data = bytearray(nf.encode_model(model))
    has_bias_offset = 4 + 4 + 4 + 8 + 4 + 4 + 4 + 4 + 4 + 4  # after the layer type id
    assert data[has_bias_offset] == 0
    data[has_bias_offset] = 2
    with pytest.raises(nf.NawaFormatError, match="has_bias"):
        nf.decode_model(bytes(data))


def test_model_truncated_and_trailing():
    data = nf.encode_model(make_model())
    with pytest.raises(nf.NawaFormatError, match="truncated"):
        nf.decode_model(data[:-2])
    with pytest.raises(nf.NawaFormatError, match="trailing"):
        nf.decode_model(data + b"\x01")


def test_bias_shape_mismatch_rejected():
    bad = nf.Model((4,), nf.Normalization(1.0, [0.0], [1.0]),
                   [nf.Linear(np.zeros((4, 3), dtype=np.float32), np.zeros(2, dtype=np.float32))])
    with pytest.raises(nf.NawaFormatError, match="bias shape"):
        nf.encode_model(bad)


def test_weight_must_be_2d():
    bad = nf.Model((4,), nf.Normalization(1.0, [0.0], [1.0]),
                   [nf.Linear(np.zeros(4, dtype=np.float32))])
    with pytest.raises(nf.NawaFormatError, match="2-D"):
        nf.encode_model(bad)


def test_invalid_normalization_rejected():
    for norm in [nf.Normalization(1.0, [0.0], [0.0]),       # std = 0
                 nf.Normalization(0.0, [0.0], [1.0]),       # pixel_scale = 0
                 nf.Normalization(1.0, [0.0, 0.0], [1.0])]:  # length mismatch
        with pytest.raises(nf.NawaFormatError):
            nf.encode_model(nf.Model((4,), norm, [nf.ReLU()]))


def test_empty_model_rejected():
    with pytest.raises(nf.NawaFormatError, match="at least one layer"):
        nf.encode_model(nf.Model((4,), nf.Normalization(1.0, [0.0], [1.0]), []))
