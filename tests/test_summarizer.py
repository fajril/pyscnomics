"""
A unit test for module helper.py
"""

import numpy as np

from pyscnomics.tools import helper


def test_summarizer():
    """
    A unit test for summarizer test.
    """

    # Case 1: First array is earlier than second array
    array1 = np.array([100, 100, 100, 100, 100])
    array2 = np.array([50, 50, 50])
    startYear1 = 2023
    startYear2 = 2025

    expected_result_1 = [100, 100, 150, 150, 150]

    result_1 = helper.summarizer(
        array1=array1, array2=array2, startYear1=startYear1, startYear2=startYear2
    )

    # Case 2: Second array is earlier than fisrt array
    array1 = np.array([50, 50, 50, 50, 50])
    array2 = np.array([100, 100, 100, 100, 100])
    startYear1 = 2025
    startYear2 = 2023

    expected_result_2 = [100, 100, 150, 150, 150, 50, 50]

    result_2 = helper.summarizer(
        array1=array1, array2=array2, startYear1=startYear1, startYear2=startYear2
    )

    # Case 3: Second array is later and shorter than first array
    array1 = np.array([100, 100, 100, 100, 100])
    array2 = np.array([100, 100, 100])
    startYear1 = 2023
    startYear2 = 2024

    expected_result_3 = [100, 200, 200, 200, 100]

    result_3 = helper.summarizer(
        array1=array1, array2=array2, startYear1=startYear1, startYear2=startYear2
    )

    # Case 4: first array is later and shorter than second array
    array1 = np.array([100, 100, 100])
    array2 = np.array([100, 100, 100, 100, 100])
    startYear1 = 2024
    startYear2 = 2023

    expected_result_4 = [100, 200, 200, 200, 100]

    result_4 = helper.summarizer(
        array1=array1, array2=array2, startYear1=startYear1, startYear2=startYear2
    )

    np.testing.assert_allclose(expected_result_1, result_1[0])
    np.testing.assert_allclose(expected_result_2, result_2[0])
    np.testing.assert_allclose(expected_result_3, result_3[0])
    np.testing.assert_allclose(expected_result_4, result_4[0])
