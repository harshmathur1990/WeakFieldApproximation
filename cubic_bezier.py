import numpy as np


# find the a & b points
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]

# evalute each cubic curve on the range [0, 1] sliced in n points
def prepare_evaluate_bezier(node_position, log_tau, edge_interp=0):
    def evaluate_bezier(nodes):

        if node_position.size >=3:
            curves = get_bezier_cubic(nodes)
            model_values = list()
            new_node_position = [np.NINF] + list(node_position) + [np.Inf]
            i = 0
            while(i < len(new_node_position) - 1):
                if (new_node_position[i] == np.NINF):

                    curve = curves[i]

                    x2 = new_node_position[i + 2]
                    x1 = new_node_position[i + 1]
                    y2 = 1
                    y1 = 0

                    ind = np.where(log_tau < x1)
                    lt = log_tau[ind]

                    if edge_interp == 0:
                        model_values += list(np.ones_like(lt) * curve(0))
                    else:
                        indl = np.where( (log_tau >= x1) & (log_tau < x2))
                        ltl = log_tau[indl]
                        t = ((y2 - y1) * (ltl - x1) / (x2 - x1)) + y1

                        a, b = np.polyfit(ltl, curve(t), 1)

                        model_values += list(a * lt + b)

                elif (new_node_position[i + 1] == np.Inf):
                    curve = curves[i - 2]

                    x2 = new_node_position[i]
                    x1 = new_node_position[i - 1]
                    y2 = 1
                    y1 = 0

                    ind = np.where(log_tau >= new_node_position[i])
                    lt = log_tau[ind]

                    if edge_interp == 0:
                        model_values += list(np.ones_like(lt) * curve(1))
                    else:
                        indl = np.where( (log_tau >= x1) & (log_tau < x2))
                        ltl = log_tau[indl]
                        t = ((y2 - y1) * (ltl - x1) / (x2 - x1)) + y1

                        a, b = np.polyfit(ltl, curve(t), 1)

                        model_values += list(a * lt + b)

                else:
                    curve = curves[i - 1]

                    x2 = new_node_position[i + 1]
                    x1 = new_node_position[i]
                    y2 = 1
                    y1 = 0

                    ind = np.where( (log_tau >=x1) & (log_tau < x2))
                    lt = log_tau[ind]

                    t = ((y2 - y1) * (lt - x1) / (x2 - x1)) + y1

                    model_values += list(curve(t))
                
                i = i + 1
            return np.array(model_values)

        else:
            return np.ones_like(log_tau) * nodes[0]
    return evaluate_bezier
