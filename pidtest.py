import numpy as np


def realistic_pid_controller_simulator_for_autoguider(value1, value2, target1, target2, a1, a2, b1, b2, kp, ki, kd, noise=0, drift=1):

    errorvoltage1 = 0
    errorvoltage2 = 0
    integral_error1 = 0
    integral_error2 = 0
    prev_error1 = 0
    prev_error2 = 0
    error_derivative1 = 0
    error_derivative2 = 0
    res_list1, res_list2 = list(), list()
    corr_list1, corr_list2 = list(), list()
    motor_count1, motor_count2 = list(), list()

    for i in range(3600 * 4):
        errorvoltage1 = target1 - value1
        errorvoltage2 = target2 - value2
        integral_error1 += errorvoltage1
        integral_error2 += errorvoltage2
        error_derivative1 = errorvoltage1 - prev_error1
        error_derivative2 = errorvoltage2 - prev_error2
        correction_voltage1 = kp * errorvoltage1 + ki * integral_error1 + kd * error_derivative1
        correction_voltage2 = kp * errorvoltage2 + ki * integral_error2 + kd * error_derivative2
        mc1 = correction_voltage1 * a1 + correction_voltage2 * a2
        mc2 = correction_voltage1 * b1 + correction_voltage2 * b2
        value1 += correction_voltage1 + drift * 4 + np.random.randint(-np.abs(noise * value1), np.abs(noise * value1))
        value2 += correction_voltage2 + drift * 4 + np.random.randint(-np.abs(noise * value2), np.abs(noise * value2))
        res_list1.append(value1)
        res_list2.append(value2)
        corr_list1.append(correction_voltage1)
        corr_list2.append(correction_voltage2)
        motor_count1.append(mc1)
        motor_count2.append(mc2)
    return res_list1, res_list2, corr_list1, corr_list2, motor_count1, motor_count2
