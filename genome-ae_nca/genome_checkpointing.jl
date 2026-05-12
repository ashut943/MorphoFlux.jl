function save_genome_model(model::GenomeCAModel, hp::GenomeHParams, path::String)
    JLD2.jldsave(path;
        encoder_w1       = Array(model.encoder_w1),
        encoder_b1       = Array(model.encoder_b1),
        encoder_mu_w     = Array(model.encoder_mu_w),
        encoder_mu_b     = Array(model.encoder_mu_b),
        encoder_logvar_w = Array(model.encoder_logvar_w),
        encoder_logvar_b = Array(model.encoder_logvar_b),
        hyper_w1         = Array(model.hyper_w1),
        hyper_b1         = Array(model.hyper_b1),
        hyper_w2         = Array(model.hyper_w2),
        hyper_b2         = Array(model.hyper_b2),
        base_theta       = Array(model.base_theta),
        channel_n        = model.channel_n,
        hidden_n         = model.hidden_n,
        target_w         = model.target_w,
        target_h         = model.target_h,
        latent_k         = model.latent_k,
        hyper_output_scale = model.hyper_output_scale,
        fire_rate        = model.fire_rate,
        hparams          = hp,
    )
end

function load_genome_model(path::String)
    d  = JLD2.load(path)
    hp = d["hparams"]
    model = GenomeCAModel(
        cu(d["encoder_w1"]),
        cu(d["encoder_b1"]),
        cu(d["encoder_mu_w"]),
        cu(d["encoder_mu_b"]),
        cu(d["encoder_logvar_w"]),
        cu(d["encoder_logvar_b"]),
        cu(d["hyper_w1"]),
        cu(d["hyper_b1"]),
        cu(d["hyper_w2"]),
        cu(d["hyper_b2"]),
        cu(d["base_theta"]),
        d["channel_n"],
        d["hidden_n"],
        d["target_w"],
        d["target_h"],
        d["latent_k"],
        d["hyper_output_scale"],
        d["fire_rate"],
    )
    return model, hp
end
