import numpy as np


def get_baseline():
    pp_stage1 = [
        0.5126,
        2.0920,
        2.4596,
        1.1221,
        0.0714,
        0.0450,
    ]  # pp_mult in fitted_parameters
    pp_stage2 = [
        1.2028,
        6.8225,
        22.3263,
        4.0448,
        0.2858,
    ]  # pp_mult_stage2 in fitted_parameters

    params = pp_stage1
    b1_choice = params[0]
    b2_sample = params[1]
    b3_sample = params[2]
    tau = params[3]
    b5_approach = params[4]
    b4_approach = params[5]

    qv = np.arange(1, 11)

    VCa = b1_choice * (qv - np.mean(qv))
    VCb = b1_choice * (np.mean(qv) - qv)
    choice_uncertainty = -((1 / (1 + np.exp(VCb - VCa)) - 0.5) ** 2)
    VSa = b2_sample + b3_sample * choice_uncertainty - b5_approach * (qv)
    VSb = b2_sample + b3_sample * choice_uncertainty

    # AA trials, big

    valsAA = np.stack([VCa, VCb + b4_approach * np.mean(qv), VSa])
    probsAA = np.exp(valsAA / tau) / np.sum(np.exp(valsAA / tau), axis=0)

    # AB trials, big

    valsAB = np.stack([VCa + b4_approach * qv, VCb, VSb])
    probsAB = np.exp(valsAB / tau) / np.sum(np.exp(valsAB / tau), axis=0)

    params = pp_stage2
    b1_choice = params[0]
    b2_sample = params[1]
    b3_sample = params[2]
    tau = params[3]
    b6_approach = params[4]

    # 2nd stage, big

    qvA = np.arange(1, 11).reshape(10, 1)
    qvB = np.arange(1, 11).reshape(1, 10)

    # value of choosing and sampling

    VCa = b1_choice * (qvA - qvB)
    VCb = b1_choice * (qvB - qvA)
    choice_uncertainty = -((1 / (1 + np.exp(VCb - VCa)) - 0.5) ** 2)
    VSa = b2_sample + b3_sample * choice_uncertainty + b6_approach * (qvA - qvB)
    VSb = b2_sample + b3_sample * choice_uncertainty + b6_approach * (qvB - qvA)

    vals2 = np.zeros((10, 10, 4))
    vals2[:, :, 0] = VCa
    vals2[:, :, 1] = VCb
    vals2[:, :, 2] = VSa
    vals2[:, :, 3] = VSb

    PsB = np.exp(vals2 / tau) / np.sum(np.exp(vals2 / tau), axis=-1, keepdims=True)

    return probsAA, probsAB, valsAA, valsAB, vals2
