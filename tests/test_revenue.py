"""
Developed by:

        - Muhammad Adhim Mulia, M.T. [1]
        - Aditya Dewanto Hartono, Dr. Eng. [1]
        - Fajril Ambia, Ph.D. [2]
        - Iskandar Fahmi, Ph.D. [3]

        Affiliations
        ____________

        [1] Lembaga Pengembangan Inovasi dan Kewirausahaan (LPIK), Institut Teknologi Bandung
            Gedung Energi R7f, Jalan Ganesha 10 Bandung 40132, Indonesia

        [2] Satuan Kerja Khusus Pelaksana Kegiatan Usaha Hulu Minyak dan Gas Bumi
            Gedung Wisma Mulia Lantai 35, Jalan Gatot Subroto 42, Jakarta 12710, Indonesia

        [3] Badan Pengelola Migas Aceh
            Jalan Stadion H. Dimurthala No.8, Kota Baru, Banda Aceh, Aceh 23125, Indonesia
"""

import numpy as np
import pytest
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import FluidType


def test_oil_revenue_for_exception_value_error():

    with pytest.raises(ValueError):

        oil_rev_calc = Lifting(
            start_year=2023,
            end_year=2027,
            lifting_rate=np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]),
            price=np.array([10 for _ in range(11)]),
            fluid_type=FluidType.OIL
        ).revenue()


def test_oil_revenue_normal_condition():

    oil_rev = np.array(
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000]
    )

    oil_rev_calc = Lifting(
        start_year=2023,
        end_year=2032,
        lifting_rate=np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        price=np.array([10 for _ in range(10)]),
        fluid_type=FluidType.OIL
    ).revenue()

    np.testing.assert_array_almost_equal(oil_rev, oil_rev_calc)


def test_oil_revenue_duration_longer_than_production_data():

    oil_rev = np.array(
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000, 0, 0, 0, 0, 0]
    )

    oil_rev_calc = Lifting(
        start_year=2023,
        end_year=2037,
        lifting_rate=np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        price=np.array([10 for _ in range(10)]),
        fluid_type=FluidType.OIL
    ).revenue()

    np.testing.assert_array_almost_equal(oil_rev, oil_rev_calc)


def test_gas_revenue_normal_condition():

    gas_rev = np.array(
        [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    )

    gas_rev_calc = Lifting(
        start_year=2023,
        end_year=2032,
        lifting_rate=np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10]),
        price=np.array([10 for _ in range(10)]),
        fluid_type=FluidType.GAS,
        ghv=np.array([1000 for _ in range(10)])
    ).revenue()
