import pytest

from bruv_app.classify import map_safetensors


def test_mapped_checkpoint_loads_into_a_model_built_on_meta(tmp_path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file

    def build():
        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
        model.register_parameter("scale", torch.nn.Parameter(torch.tensor(4.6)))  # 0-d, like logit_scale
        return model

    torch.manual_seed(0)
    source = build()
    path = tmp_path / "w.safetensors"
    save_file(source.state_dict(), str(path))

    weights, mapping = map_safetensors(path)
    with torch.device("meta"):
        model = build()  # nothing allocated
    model.load_state_dict(weights, strict=True, assign=True)
    assert not any(p.is_meta for p in model.parameters())
    for key, value in source.state_dict().items():
        assert torch.equal(model.state_dict()[key], value), key
    x = torch.randn(2, 4)
    with torch.no_grad():
        assert torch.equal(model(x), source(x))
    del model, weights
    mapping.close()
