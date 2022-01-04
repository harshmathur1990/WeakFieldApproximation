def calculate_lande_g_eff(g1, g2, j1, j2):
    j_d = calculate_d(j1, j2)

    g_s = calculate_g_s(g1, g2)

    g_d = calculate_g_d(g1, g2)

    return 0.5 * g_s + 0.25 * g_d * j_d


def calculate_lande_g_factor_for_level(j, l, s):
    return 1 + (
        0.5 * (
            (
                j * (j + 1) + s * (s + 1) - l * (l + 1)
            ) / (
                j * (j + 1)
            )
        )
    )


def calculate_s(j1, j2):
    return j1 * (j1 + 1) + j2 * (j2 + 1)


def calculate_d(j1, j2):
    return j1 * (j1 + 1) - j2 * (j2 + 1)


def calculate_g_s(g1, g2):
    return g1 + g2


def calculate_g_d(g1, g2):
    return g1 - g2


def calculate_delta(g_d, j_s, j_d):
    return g_d**2 * (16 * j_s - 7 * j_d**2 - 4) / 80


def calculate_g_eff_transverse(g1, g2, j1, j2):
    g_eff = calculate_lande_g_eff(g1, g2, j1, j2)

    j_s = calculate_s(j1, j2)

    j_d = calculate_d(j1, j2)

    g_d = calculate_g_d(g1, g2)

    delta = calculate_delta(g_d, j_s, j_d)

    return g_eff**2 - delta


def calculate_g_1_2(g_s, g_d, j_s, j_d):
    return (
        (j_d**2 + 2 * j_s - 3) * g_d**2 + 5 * j_d * g_s * g_d + 5 * g_s**2
    ) / 20


def calculate_g_0_2(g_d, j_s, j_d):
    return (3 * j_s - j_d**2 - 2) * g_d**2 / 10


def calculate_g_eff_transverse_alternate(g1, g2, j1, j2):
    j_s = calculate_s(j1, j2)

    j_d = calculate_d(j1, j2)

    g_s = calculate_g_s(g1, g2)

    g_d = calculate_g_d(g1, g2)

    g_1_2 = calculate_g_1_2(g_s, g_d, j_s, j_d)

    g_0_2 = calculate_g_0_2(g_d, j_s, j_d)

    return g_1_2 - g_0_2


def get_lande_g_eff_transverse(j1, l1, s1, j2, l2, s2):
    g1 = calculate_lande_g_factor_for_level(j1, l1, s1)

    g2 = calculate_lande_g_factor_for_level(j2, l2, s2)

    return calculate_g_eff_transverse_alternate(g1, g2, j1, j2)
