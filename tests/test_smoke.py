
def test_forward():
    import torch
    from ser_transformer.models.transformer_ser import ImprovedTransformerSER
    model = ImprovedTransformerSER(147, 8, 256, 2, 4)
    x = torch.randn(2,147,300)
    out,_ = model(x)
    assert out.shape == (2,4)
