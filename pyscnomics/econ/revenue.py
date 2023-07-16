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
from pyscnomics.econ.costs import FluidType
from dataclasses import dataclass, field


@dataclass
class Lifting:
    start_year: int
    end_year: int
    lifting_rate: np.ndarray
    price: np.ndarray
    fluid_type: FluidType
    ghv: np.ndarray = field(default=None, repr=False)
    prod_rate: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):

        if self.prod_rate is None:
            self.prod_rate = self.lifting_rate.copy()

        if self.ghv is None:
            self.ghv = np.ones(len(self.prod_rate))

        arr_length = self.lifting_rate.shape[0]

        if not all(
                len(arr) == arr_length
                for arr in [
                    self.price,
                    self.ghv,
                    self.prod_rate
                ]
        ):
            raise ValueError(
                f'Inequal length of array: lifting_rate: {len(self.lifting_rate)},'
                f' ghv: {len(self.ghv)},'
                f' production: {len(self.prod_rate)}'
            )

        # Define an attribute depicting the project duration
        if self.end_year > self.start_year:
            self.project_duration = self.end_year - self.start_year + 1

        else:
            raise ValueError(
                f"start year {self.start_year} is after the end year: {self.end_year}"
            )

        # Specify an error condition when project duration is less than the length of production data
        if self.project_duration < len(self.prod_rate):
            raise ValueError(
                f'Length of project duration: ({self.project_duration})'
                f' is less than the length of production data: ({len(self.prod_rate)})'
            )
        
    def revenue(self):

        rev = self.lifting_rate * self.price * self.ghv

        # When project duration is longer than the length of production data,
        # assign the revenue of the suplementary years with zeros
        if self.project_duration > len(self.prod_rate):
            add_zeros = np.zeros(int(self.project_duration - len(self.prod_rate)))
            rev = np.concatenate((rev, add_zeros))

        return rev
