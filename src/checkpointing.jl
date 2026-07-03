function save_model(model::CAModel, hp::HParams, path::String)
    JLD2.jldsave(path;
        dense1_w = Array(model.dense1_w),
        dense1_b = Array(model.dense1_b),
        dense2_w = Array(model.dense2_w),
        dense2_b = Array(model.dense2_b),
        channel_n = model.channel_n,
        fire_rate = model.fire_rate,
        visible_channels = model.visible_channels,
        hparams  = hp,
    )
end

function load_model(path::String)
    d = JLD2.load(path)
    hp = d["hparams"]
    model = CAModel(
        cu(d["dense1_w"]),
        cu(d["dense1_b"]),
        cu(d["dense2_w"]),
        cu(d["dense2_b"]),
        d["channel_n"],
        d["fire_rate"],
        get(d, "visible_channels", 4),
    )
    return model, hp
end
