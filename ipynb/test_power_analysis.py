#!/usr/bin/env python3
"""
Power Analysis Verification Script
Tests that the corrected functions produce expected results
"""

import subprocess
import sys

# Install statsmodels if needed
try:
    from statsmodels.stats.power import FTestPower
except ImportError:
    print("Installing statsmodels...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "statsmodels", "--break-system-packages", "-q"])
    from statsmodels.stats.power import FTestPower

import numpy as np

def calculate_regression_power(n, k, f2, alpha=0.05):
    """
    Calculate statistical power for multiple regression using F-test.
    CORRECTED VERSION - uses df_num and df_denom parameters
    """
    power_analysis = FTestPower()
    df_num = k                # numerator df (number of predictors)
    df_denom = n - k - 1      # denominator df (residual)
    
    power = power_analysis.solve_power(
        effect_size=f2,
        df_num=df_num,
        df_denom=df_denom,
        alpha=alpha,
        power=None            # Solving for power
    )
    return power

def minimum_detectable_effect(n, k, power=0.80, alpha=0.05):
    """
    Calculate minimum detectable Cohen's f² for given sample size and power.
    CORRECTED VERSION - uses df_num and df_denom parameters
    """
    power_analysis = FTestPower()
    df_num = k
    df_denom = n - k - 1
    
    f2_mde = power_analysis.solve_power(
        effect_size=None,     # Solving for effect size
        df_num=df_num,
        df_denom=df_denom,
        alpha=alpha,
        power=power
    )
    return f2_mde

def cohens_f2_from_r2(r2):
    """Convert R² to Cohen's f²"""
    return r2 / (1 - r2)

def r2_from_cohens_f2(f2):
    """Convert Cohen's f² to R²"""
    return f2 / (1 + f2)

# Study parameters
CURRENT_N = 154
PREVIOUS_N = 75
N_FEATURES = 9
ALPHA = 0.05
TARGET_POWER = 0.80

print("="*70)
print("POWER ANALYSIS VERIFICATION TEST")
print("="*70)
print(f"\nStudy parameters:")
print(f"  Current sample: N = {CURRENT_N}")
print(f"  Previous sample: N = {PREVIOUS_N}")
print(f"  Number of predictors: k = {N_FEATURES}")
print(f"  Significance level: α = {ALPHA}")
print(f"  Target power: {TARGET_POWER}")

# Test 1: Power for different effect sizes
print("\n" + "="*70)
print("TEST 1: Power Calculations for Different Effect Sizes")
print("="*70)

test_r2_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]

print(f"\n{'R²':<8} {'f²':<8} {'Power (N=75)':<15} {'Power (N=154)':<15} {'Status'}")
print("-"*70)

all_pass = True
for r2 in test_r2_values:
    f2 = cohens_f2_from_r2(r2)
    power_prev = calculate_regression_power(PREVIOUS_N, N_FEATURES, f2, ALPHA)
    power_curr = calculate_regression_power(CURRENT_N, N_FEATURES, f2, ALPHA)
    
    # Power should increase with sample size
    status = "PASS" if power_curr > power_prev else "FAIL"
    if status == "FAIL":
        all_pass = False
    
    print(f"{r2:<8.3f} {f2:<8.3f} {power_prev:<15.4f} {power_curr:<15.4f} {status}")

print(f"\nTest 1 Result: {'PASS - Power increases with sample size' if all_pass else 'FAIL'}")

# Test 2: Minimum detectable effect
print("\n" + "="*70)
print("TEST 2: Minimum Detectable Effect Size")
print("="*70)

mde_prev_f2 = minimum_detectable_effect(PREVIOUS_N, N_FEATURES, TARGET_POWER, ALPHA)
mde_curr_f2 = minimum_detectable_effect(CURRENT_N, N_FEATURES, TARGET_POWER, ALPHA)

mde_prev_r2 = r2_from_cohens_f2(mde_prev_f2)
mde_curr_r2 = r2_from_cohens_f2(mde_curr_f2)

print(f"\n{'Sample':<20} {'N':<10} {'MDE (R²)':<15} {'MDE (f²)':<15}")
print("-"*70)
print(f"{'Previous':<20} {PREVIOUS_N:<10} {mde_prev_r2:<15.4f} {mde_prev_f2:<15.4f}")
print(f"{'Current':<20} {CURRENT_N:<10} {mde_curr_r2:<15.4f} {mde_curr_f2:<15.4f}")

# MDE should decrease with sample size
test2_pass = mde_curr_r2 < mde_prev_r2
print(f"\nTest 2 Result: {'PASS - MDE decreases with sample size' if test2_pass else 'FAIL'}")

# Test 3: Verify power at MDE equals target power
print("\n" + "="*70)
print("TEST 3: Verification - Power at MDE should equal target power")
print("="*70)

power_at_mde_curr = calculate_regression_power(CURRENT_N, N_FEATURES, mde_curr_f2, ALPHA)
power_at_mde_prev = calculate_regression_power(PREVIOUS_N, N_FEATURES, mde_prev_f2, ALPHA)

print(f"\n{'Sample':<20} {'N':<10} {'MDE (f²)':<15} {'Power at MDE':<15} {'Target':<10}")
print("-"*70)
print(f"{'Previous':<20} {PREVIOUS_N:<10} {mde_prev_f2:<15.4f} {power_at_mde_prev:<15.4f} {TARGET_POWER:<10.2f}")
print(f"{'Current':<20} {CURRENT_N:<10} {mde_curr_f2:<15.4f} {power_at_mde_curr:<15.4f} {TARGET_POWER:<10.2f}")

# Allow small numerical error (within 0.001)
test3_pass = (abs(power_at_mde_curr - TARGET_POWER) < 0.001 and 
              abs(power_at_mde_prev - TARGET_POWER) < 0.001)
print(f"\nTest 3 Result: {'PASS - Powers match target within tolerance' if test3_pass else 'FAIL'}")

# Test 4: Expected values validation
print("\n" + "="*70)
print("TEST 4: Validation Against Expected Values")
print("="*70)

expected_values = {
    0.15: {'N75': 0.49, 'N154': 0.95},  # Medium effect
    0.25: {'N75': 0.81, 'N154': 0.99},  # Large effect
}

print(f"\n{'R²':<8} {'Sample':<10} {'Expected':<12} {'Calculated':<12} {'Match'}")
print("-"*70)

test4_pass = True
for r2, expected in expected_values.items():
    f2 = cohens_f2_from_r2(r2)
    
    # Test N=75
    calc_75 = calculate_regression_power(PREVIOUS_N, N_FEATURES, f2, ALPHA)
    match_75 = abs(calc_75 - expected['N75']) < 0.02  # Within 2% tolerance
    if not match_75:
        test4_pass = False
    print(f"{r2:<8.2f} {'N=75':<10} {expected['N75']:<12.2f} {calc_75:<12.4f} {'PASS' if match_75 else 'FAIL'}")
    
    # Test N=154
    calc_154 = calculate_regression_power(CURRENT_N, N_FEATURES, f2, ALPHA)
    match_154 = abs(calc_154 - expected['N154']) < 0.02
    if not match_154:
        test4_pass = False
    print(f"{r2:<8.2f} {'N=154':<10} {expected['N154']:<12.2f} {calc_154:<12.4f} {'PASS' if match_154 else 'FAIL'}")

print(f"\nTest 4 Result: {'PASS - Values match expected within 2%' if test4_pass else 'FAIL'}")

# Overall summary
print("\n" + "="*70)
print("OVERALL TEST SUMMARY")
print("="*70)

all_tests_pass = all_pass and test2_pass and test3_pass and test4_pass

print(f"\nTest 1 (Power increases with N): {'PASS' if all_pass else 'FAIL'}")
print(f"Test 2 (MDE decreases with N): {'PASS' if test2_pass else 'FAIL'}")
print(f"Test 3 (Power at MDE = target): {'PASS' if test3_pass else 'FAIL'}")
print(f"Test 4 (Expected values match): {'PASS' if test4_pass else 'FAIL'}")

print(f"\n{'='*70}")
if all_tests_pass:
    print("ALL TESTS PASSED - Functions are working correctly")
    print("The corrected notebook is ready to use.")
else:
    print("SOME TESTS FAILED - Review function implementations")

print("="*70)
